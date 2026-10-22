import AppKit
import SwiftUI

@main
struct LocalAIApp: App {
    @StateObject private var controller = ServiceController()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(controller)
                .frame(minWidth: 760, minHeight: 620)
        }
    }
}

private enum Runtime: String, CaseIterable, Identifiable {
    case mlx
    case ollama
    case llamaCpp = "llama_cpp"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mlx: "MLX"
        case .ollama: "Ollama"
        case .llamaCpp: "llama.cpp server"
        }
    }
}

private struct AppSettings: Codable {
    var provider = Runtime.mlx.rawValue
    var model = ""
    var ollama_url = "http://127.0.0.1:11434"
    var llama_cpp_url = "http://127.0.0.1:8080"
}

private struct SearchResponse: Decodable {
    let data: [HuggingFaceModel]
}

private struct HuggingFaceModel: Decodable, Identifiable {
    let id: String
    let downloads: Int
    let likes: Int
    let pipeline_tag: String?
}

private struct ModelListResponse: Decodable {
    let data: [InstalledModel]
}

private struct InstalledModel: Decodable, Identifiable {
    let id: String
    let path: String
}

private struct DownloadResponse: Decodable {
    let repo_id: String
    let path: String
}

private struct ChatRequest: Encodable {
    let model: String
    let messages: [ChatMessage]
    let stream = false
}

private struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    let role: String
    let content: String

    enum CodingKeys: String, CodingKey {
        case role
        case content
    }
}

private struct ChatResponse: Decodable {
    let choices: [Choice]

    struct Choice: Decodable {
        let message: ChatMessage
    }
}

@MainActor
private final class ServiceController: ObservableObject {
    @Published var isRunning = false
    @Published var status = "Serviço parado"
    @Published var settings = AppSettings()
    @Published var searchQuery = ""
    @Published var searchResults: [HuggingFaceModel] = []
    @Published var installedModels: [InstalledModel] = []
    @Published var messages: [ChatMessage] = []
    @Published var prompt = ""
    @Published var isBusy = false
    @Published var isDownloading = false

    private var process: Process?
    private let baseURL = URL(string: "http://127.0.0.1:11435")!

    func startService() {
        guard process?.isRunning != true else { return }
        guard let backendPath = findBackendPath() else {
            status = "Backend Python não encontrado. Configure LOCAL_AI_BACKEND_PATH."
            return
        }

        let launchedProcess = Process()
        launchedProcess.executableURL = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["LOCAL_AI_PYTHON"] ?? "/usr/bin/env"
        )
        if launchedProcess.executableURL?.lastPathComponent == "env" {
            launchedProcess.arguments = ["python3", "-m", "local_ai", "--host", "127.0.0.1", "--port", "11435"]
        } else {
            launchedProcess.arguments = ["-m", "local_ai", "--host", "127.0.0.1", "--port", "11435"]
        }
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = backendPath.path
        launchedProcess.environment = environment
        launchedProcess.currentDirectoryURL = backendPath

        let output = Pipe()
        launchedProcess.standardOutput = output
        launchedProcess.standardError = output
        output.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            FileHandle.standardError.write(Data(text.utf8))
        }
        launchedProcess.terminationHandler = { [weak self] endedProcess in
            Task { @MainActor in
                self?.isRunning = false
                self?.status = "Serviço encerrado (código \(endedProcess.terminationStatus))"
            }
        }

        do {
            try launchedProcess.run()
            process = launchedProcess
            status = "Iniciando serviço local…"
            Task { await waitForService() }
        } catch {
            status = "Falha ao iniciar o Python: \(error.localizedDescription)"
        }
    }

    func stopService() {
        guard let process, process.isRunning else {
            isRunning = false
            status = "Serviço parado"
            return
        }
        process.terminate()
        self.process = nil
        isRunning = false
        status = "Serviço parado"
    }

    func saveSettings() async {
        do {
            settings = try await sendJSON(settings, to: "/api/settings", response: AppSettings.self)
            status = "Configuração salva"
        } catch {
            status = error.localizedDescription
        }
    }

    func loadSettings() async {
        do {
            settings = try await get("/api/settings", as: AppSettings.self)
        } catch {
            status = "Falha ao carregar configuração: \(error.localizedDescription)"
        }
    }

    func refreshModels() async {
        do {
            let response = try await get("/api/models", as: ModelListResponse.self)
            installedModels = response.data
        } catch {
            status = "Falha ao carregar modelos locais: \(error.localizedDescription)"
        }
    }

    func searchHuggingFace() async {
        guard !searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            status = "Digite um termo para pesquisar no Hugging Face."
            return
        }
        do {
            guard var components = URLComponents(
                url: baseURL.appendingPathComponent("api/huggingface/search"),
                resolvingAgainstBaseURL: false
            ) else {
                throw APIError.message("Não foi possível criar a URL de pesquisa.")
            }
            components.queryItems = [URLQueryItem(name: "q", value: searchQuery)]
            guard let searchURL = components.url else {
                throw APIError.message("Não foi possível criar a URL de pesquisa.")
            }
            searchResults = try await request(
                searchURL,
                method: "GET",
                body: nil,
                as: SearchResponse.self
            ).data
            status = "\(searchResults.count) modelos encontrados"
        } catch {
            status = error.localizedDescription
        }
    }

    func download(_ model: HuggingFaceModel) async {
        isBusy = true
        isDownloading = true
        status = "Baixando \(model.id)…"
        defer {
            isBusy = false
            isDownloading = false
        }
        do {
            let result = try await sendJSON(
                ["repo_id": model.id],
                to: "/api/huggingface/download",
                response: DownloadResponse.self
            )
            settings.model = result.path
            status = "Download concluído: \(result.repo_id). Salve a configuração para ativar."
            await refreshModels()
        } catch {
            status = error.localizedDescription
        }
    }

    func useInstalledModel(_ model: InstalledModel) {
        settings.model = model.path
    }

    func sendMessage() async {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isBusy else { return }
        guard isRunning else {
            status = "Inicie o serviço antes de enviar uma mensagem."
            return
        }
        isBusy = true
        prompt = ""
        messages.append(ChatMessage(role: "user", content: text))
        defer { isBusy = false }
        do {
            let response = try await sendJSON(
                ChatRequest(model: settings.model, messages: messages),
                to: "/v1/chat/completions",
                response: ChatResponse.self
            )
            if let answer = response.choices.first?.message.content {
                messages.append(ChatMessage(role: "assistant", content: answer))
            }
        } catch {
            status = error.localizedDescription
        }
    }

    private func waitForService() async {
        for _ in 0..<40 {
            do {
                let _: [String: String] = try await get("/health", as: [String: String].self)
                guard process?.isRunning == true else {
                    throw APIError.message("O processo do serviço foi encerrado.")
                }
                isRunning = true
                status = "API local ativa em http://127.0.0.1:11435"
                await loadSettings()
                await refreshModels()
                return
            } catch {
                try? await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        status = "O serviço não respondeu. Verifique o Python e as dependências."
    }

    private func findBackendPath() -> URL? {
        if let configured = ProcessInfo.processInfo.environment["LOCAL_AI_BACKEND_PATH"] {
            let path = URL(fileURLWithPath: configured)
            if FileManager.default.fileExists(atPath: path.appendingPathComponent("local_ai").path) {
                return path
            }
        }
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("Backend"),
           FileManager.default.fileExists(atPath: bundled.appendingPathComponent("local_ai").path) {
            return bundled
        }
        let sourceTree = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("src")
        if FileManager.default.fileExists(atPath: sourceTree.appendingPathComponent("local_ai").path) {
            return sourceTree
        }
        return nil
    }

    private func get<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        try await request(path, method: "GET", body: Optional<Data>.none, as: type)
    }

    private func sendJSON<Body: Encodable, Response: Decodable>(
        _ body: Body,
        to path: String,
        response: Response.Type
    ) async throws -> Response {
        try await request(path, method: "POST", body: JSONEncoder().encode(body), as: response)
    }

    private func request<T: Decodable>(
        _ path: String,
        method: String,
        body: Data?,
        as type: T.Type
    ) async throws -> T {
        guard let requestURL = URL(string: path, relativeTo: baseURL)?.absoluteURL else {
            throw APIError.message("URL inválida para o serviço local.")
        }
        return try await request(requestURL, method: method, body: body, as: type)
    }

    private func request<T: Decodable>(
        _ url: URL,
        method: String,
        body: Data?,
        as type: T.Type
    ) async throws -> T {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        if body != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.message("Resposta inválida do serviço local.")
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let payload = try? JSONDecoder().decode(APIErrorResponse.self, from: data)
            throw APIError.message(payload?.error.message ?? "Erro HTTP \(httpResponse.statusCode)")
        }
        return try JSONDecoder().decode(type, from: data)
    }
}

private struct APIErrorResponse: Decodable {
    struct APIError: Decodable {
        let message: String
    }
    let error: APIError
}

private enum APIError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        if case .message(let message) = self { return message }
        return nil
    }
}

private enum WorkspacePage: String, CaseIterable, Identifiable {
    case overview
    case models
    case chat
    case vscode

    var id: String { rawValue }

    var title: String {
        switch self {
        case .overview: "Visão geral"
        case .models: "Modelos"
        case .chat: "Playground"
        case .vscode: "VS Code"
        }
    }

    var icon: String {
        switch self {
        case .overview: "square.grid.2x2"
        case .models: "shippingbox"
        case .chat: "bubble.left.and.bubble.right"
        case .vscode: "chevron.left.forwardslash.chevron.right"
        }
    }
}

private struct ContentView: View {
    @EnvironmentObject private var controller: ServiceController
    @State private var selectedPage: WorkspacePage? = .overview

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 260)
        } detail: {
            VStack(spacing: 0) {
                header
                Divider()
                ScrollView {
                    pageContent
                        .padding(28)
                        .frame(maxWidth: 1040, alignment: .leading)
                        .frame(maxWidth: .infinity)
                }
                .background(Color(nsColor: .windowBackgroundColor))
                Divider()
                statusBar
            }
        }
        .tint(Color.accentColor)
        .onDisappear {
            controller.stopService()
        }
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack(spacing: 11) {
                Image(systemName: "cpu.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(
                        LinearGradient(
                            colors: [Color.accentColor, Color.accentColor.opacity(0.72)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        in: RoundedRectangle(cornerRadius: 11, style: .continuous)
                    )
                VStack(alignment: .leading, spacing: 2) {
                    Text("Local AI").font(.headline)
                    Text("MLX WORKSPACE")
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                        .tracking(1.1)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 18)
            .padding(.top, 20)
            .padding(.bottom, 18)

            List(selection: $selectedPage) {
                Section("WORKSPACE") {
                    ForEach([WorkspacePage.overview, .models, .chat]) { page in
                        Label(page.title, systemImage: page.icon)
                            .tag(page)
                    }
                }
                Section("INTEGRAÇÕES") {
                    Label(WorkspacePage.vscode.title, systemImage: WorkspacePage.vscode.icon)
                        .tag(WorkspacePage.vscode)
                }
            }
            .listStyle(.sidebar)

            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(controller.isRunning ? Color.green : Color.secondary.opacity(0.55))
                        .frame(width: 8, height: 8)
                    Text(controller.isRunning ? "Serviço local ativo" : "Serviço desconectado")
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                }
                Text(controller.settings.model.isEmpty ? "Nenhum modelo selecionado" : controller.settings.model)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Button {
                    controller.isRunning ? controller.stopService() : controller.startService()
                } label: {
                    Label(
                        controller.isRunning ? "Parar serviço" : "Iniciar serviço",
                        systemImage: controller.isRunning ? "stop.fill" : "play.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            .padding(14)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .padding(12)
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private var pageContent: some View {
        switch selectedPage ?? .overview {
        case .overview:
            overviewPage
        case .models:
            modelsPage
        case .chat:
            chatPage
        case .vscode:
            vscodePage
        }
    }

    private var overviewPage: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Seu laboratório de IA, no Mac.")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .tracking(-0.7)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Execute modelos localmente com MLX e conecte suas ferramentas de desenvolvimento.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Image(systemName: "sparkles")
                    .font(.system(size: 25, weight: .medium))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 58, height: 58)
                    .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 18))
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), spacing: 12)], spacing: 12) {
                statusMetric(
                    title: "SERVIÇO",
                    value: controller.isRunning ? "Pronto" : "Desconectado",
                    icon: controller.isRunning ? "bolt.horizontal.circle.fill" : "bolt.slash.circle",
                    tint: controller.isRunning ? .green : .secondary
                )
                statusMetric(title: "MOTOR", value: runtimeName, icon: "cpu", tint: .accentColor)
                statusMetric(
                    title: "MODELOS",
                    value: "\(controller.installedModels.count) disponíveis",
                    icon: "shippingbox",
                    tint: .accentColor
                )
            }

            VStack(alignment: .leading, spacing: 14) {
                sectionHeading("Ambiente de execução", subtitle: "Escolha o runtime e o modelo para este workspace.")
                settingsSection
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 14)], spacing: 14) {
                quickLink(
                    title: "Explorar modelos",
                    subtitle: "Pesquise no Hugging Face",
                    icon: "magnifyingglass",
                    page: .models
                )
                quickLink(
                    title: "Conectar ao VS Code",
                    subtitle: "Endpoint OpenAI local",
                    icon: WorkspacePage.vscode.icon,
                    page: .vscode
                )
            }
        }
    }

    private var modelsPage: some View {
        VStack(alignment: .leading, spacing: 22) {
            pageHeading(
                "Modelos",
                subtitle: "Configure o runtime e encontre modelos compatíveis com seu ambiente."
            )
            settingsSection
            huggingFaceSection
        }
    }

    private var chatPage: some View {
        VStack(alignment: .leading, spacing: 22) {
            pageHeading(
                "Playground",
                subtitle: "Valide seu modelo local antes de conectá-lo ao fluxo de desenvolvimento."
            )
            chatSection
        }
    }

    private var vscodePage: some View {
        VStack(alignment: .leading, spacing: 22) {
            pageHeading(
                "Integração com VS Code",
                subtitle: "Use o serviço local em extensões que aceitam endpoints compatíveis com OpenAI."
            )

            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 14) {
                    Image(systemName: "arrow.left.arrow.right")
                        .font(.title2)
                        .foregroundStyle(Color.accentColor)
                        .frame(width: 46, height: 46)
                        .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 14))
                    VStack(alignment: .leading, spacing: 5) {
                        Text(controller.isRunning ? "API pronta para conectar" : "Inicie o serviço para conectar")
                            .font(.headline)
                        Text("O endpoint fica disponível apenas neste computador.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(controller.isRunning ? "Serviço ativo" : "Iniciar serviço") {
                        controller.startService()
                    }
                    .disabled(controller.isRunning)
                }

                Divider()
                connectionValue(title: "Base URL", value: "http://127.0.0.1:11435/v1", symbol: "link")
                connectionValue(
                    title: "Modelo",
                    value: controller.settings.model.isEmpty ? "Configure um modelo na seção Modelos" : controller.settings.model,
                    symbol: "shippingbox"
                )
                connectionValue(title: "Chave de API", value: "Não necessária", symbol: "key")
            }
            .padding(20)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

            VStack(alignment: .leading, spacing: 10) {
                Label("Como conectar", systemImage: "list.number")
                    .font(.headline)
                integrationStep("1", "Inicie o serviço local usando o botão acima.")
                integrationStep("2", "Na extensão do VS Code, escolha um provedor compatível com OpenAI.")
                integrationStep("3", "Informe a Base URL acima e use o modelo selecionado neste app.")
                Text("A API é vinculada a 127.0.0.1. Não exponha a porta do serviço à rede.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.accentColor.opacity(0.055), in: RoundedRectangle(cornerRadius: 16))

        }
    }

    private var runtimeName: String {
        Runtime(rawValue: controller.settings.provider)?.title ?? controller.settings.provider
    }

    private var header: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text("LOCAL AI  /  WORKSPACE")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .tracking(1.1)
                    .foregroundStyle(.secondary)
                Text((selectedPage ?? .overview).title)
                    .font(.title3.weight(.semibold))
            }
            Spacer()
            HStack(spacing: 7) {
                Circle()
                    .fill(controller.isRunning ? Color.green : Color.secondary.opacity(0.55))
                    .frame(width: 7, height: 7)
                Text(controller.isRunning ? "Conectado" : "Offline")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.quaternary, in: Capsule())
            Button {
                controller.isRunning ? controller.stopService() : controller.startService()
            } label: {
                Label(
                    controller.isRunning ? "Parar" : "Iniciar",
                    systemImage: controller.isRunning ? "stop.fill" : "play.fill"
                )
            }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            Image(systemName: controller.isRunning ? "checkmark.circle.fill" : "info.circle")
                .foregroundStyle(controller.isRunning ? Color.green : Color.secondary)
            Text(controller.status)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
            Text("LOCAL")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(1)
                .foregroundStyle(.secondary)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 20)
        .padding(.vertical, 9)
    }

    private func pageHeading(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .tracking(-0.5)
            Text(subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func sectionHeading(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.headline)
            Text(subtitle).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func statusMetric(title: String, value: String, icon: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(tint)
            Text(title)
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(1)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(15)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
    }

    private func quickLink(title: String, subtitle: String, icon: String, page: WorkspacePage) -> some View {
        Button {
            selectedPage = page
        } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.headline)
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 38, height: 38)
                    .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 11))
                VStack(alignment: .leading, spacing: 3) {
                    Text(title).font(.subheadline.weight(.semibold))
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func connectionValue(title: String, value: String, symbol: String) -> some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: symbol)
                .foregroundStyle(.secondary)
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 4)
            if title == "Base URL" {
                Button("Copiar") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(value, forType: .string)
                    controller.status = "Base URL copiada"
                }
                .controlSize(.small)
            }
        }
    }

    private func integrationStep(_ number: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(number)
                .font(.caption.weight(.bold))
                .foregroundStyle(Color.accentColor)
                .frame(width: 22, height: 22)
                .background(Color.accentColor.opacity(0.1), in: Circle())
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var settingsSection: some View {
        GroupBox("Motor e modelo") {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Motor", selection: $controller.settings.provider) {
                    ForEach(Runtime.allCases) { runtime in
                        Text(runtime.title).tag(runtime.rawValue)
                    }
                }
                TextField("Modelo (ID Hugging Face, caminho local ou nome no motor)", text: $controller.settings.model)
                    .textFieldStyle(.roundedBorder)
                if controller.settings.provider == Runtime.ollama.rawValue {
                    TextField("URL do Ollama", text: $controller.settings.ollama_url)
                        .textFieldStyle(.roundedBorder)
                    Text("Use o nome de um modelo já instalado no Ollama.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if controller.settings.provider == Runtime.llamaCpp.rawValue {
                    TextField("URL do servidor llama.cpp", text: $controller.settings.llama_cpp_url)
                        .textFieldStyle(.roundedBorder)
                    Text("Inicie o llama-server separadamente e informe o nome servido.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("MLX roda modelos compatíveis no Apple Silicon. Baixe um modelo abaixo ou informe um caminho local.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                HStack {
                    Button("Salvar configuração") {
                        Task { await controller.saveSettings() }
                    }
                    Button("Atualizar modelos baixados") {
                        Task { await controller.refreshModels() }
                    }
                    Spacer()
                }
                if !controller.installedModels.isEmpty {
                    Picker("Modelos baixados", selection: Binding(
                        get: { controller.installedModels.first(where: { $0.path == controller.settings.model })?.path ?? "" },
                        set: { path in
                            if let model = controller.installedModels.first(where: { $0.path == path }) {
                                controller.useInstalledModel(model)
                            }
                        }
                    )) {
                        Text("Selecione…").tag("")
                        ForEach(controller.installedModels) { model in
                            Text(model.id).tag(model.path)
                        }
                    }
                }
            }
            .padding(.top, 6)
        }
    }

    private var huggingFaceSection: some View {
        GroupBox("Hugging Face") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    TextField("Pesquisar modelos (ex.: mlx-community)", text: $controller.searchQuery)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await controller.searchHuggingFace() } }
                    Button("Pesquisar") {
                        Task { await controller.searchHuggingFace() }
                    }
                    .disabled(!controller.isRunning || controller.isBusy)
                }
                if controller.isDownloading {
                    ProgressView("Baixando modelo do Hugging Face…")
                }
                if !controller.searchResults.isEmpty {
                    ForEach(controller.searchResults) { model in
                        HStack(alignment: .center) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(model.id).font(.callout.weight(.medium))
                                Text("\(model.downloads.formatted()) downloads · \(model.pipeline_tag ?? "tipo não informado")")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Baixar") {
                                Task { await controller.download(model) }
                            }
                            .disabled(!controller.isRunning || controller.isBusy)
                        }
                    }
                }
            }
            .padding(.top, 6)
        }
    }

    private var chatSection: some View {
        GroupBox("Teste de conversa") {
            VStack(alignment: .leading, spacing: 12) {
                if controller.messages.isEmpty {
                    Text("As mensagens são processadas pelo motor selecionado.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(controller.messages) { message in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(message.role == "user" ? "Você" : "Modelo")
                                .font(.caption.bold())
                                .foregroundStyle(.secondary)
                            Text(message.content)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .padding(10)
                        .background(message.role == "user" ? Color.blue.opacity(0.08) : Color.gray.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
                HStack(alignment: .bottom) {
                    TextField("Escreva uma mensagem…", text: $controller.prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await controller.sendMessage() } }
                    Button("Enviar") {
                        Task { await controller.sendMessage() }
                    }
                    .disabled(!controller.isRunning || controller.isBusy)
                }
            }
            .padding(.top, 6)
        }
    }

}
