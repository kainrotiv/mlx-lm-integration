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

    private var process: Process?
    private let baseURL = URL(string: "http://127.0.0.1:11435")!

    init() {
        Task {
            await loadSettings()
            await refreshModels()
        }
    }

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
            status = "Inicie o serviço para carregar as configurações."
        }
    }

    func refreshModels() async {
        do {
            let response = try await get("/api/models", as: ModelListResponse.self)
            installedModels = response.data
        } catch {
            installedModels = []
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
        defer { isBusy = false }
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

private struct ContentView: View {
    @EnvironmentObject private var controller: ServiceController

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    settingsSection
                    huggingFaceSection
                    chatSection
                    ideSection
                }
                .padding(24)
                .frame(maxWidth: 900, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            Divider()
            Text(controller.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
        }
        .onDisappear {
            controller.stopService()
        }
    }

    private var header: some View {
        HStack {
            Image(systemName: "cpu")
                .font(.title2)
            Text("Local AI")
                .font(.title2.bold())
            Spacer()
            Circle()
                .fill(controller.isRunning ? .green : .gray)
                .frame(width: 9, height: 9)
            Text(controller.isRunning ? "Ativo" : "Parado")
                .foregroundStyle(.secondary)
            Button(controller.isRunning ? "Parar" : "Iniciar") {
                controller.isRunning ? controller.stopService() : controller.startService()
            }
            .keyboardShortcut("r", modifiers: [.command, .shift])
        }
        .padding(18)
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

    private var ideSection: some View {
        GroupBox("Conectar uma IDE") {
            VStack(alignment: .leading, spacing: 8) {
                Text("API compatível com OpenAI (disponível somente neste computador):")
                Label("Base URL: http://127.0.0.1:11435/v1", systemImage: "link")
                Label("Modelo: \(controller.settings.model.isEmpty ? "<modelo configurado no app>" : controller.settings.model)", systemImage: "shippingbox")
                Text("No VS Code, use uma extensão que permita configurar um endpoint OpenAI personalizado. Não exponha essa porta à rede.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 6)
        }
    }
}
