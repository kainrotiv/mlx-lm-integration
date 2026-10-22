# Local AI

Aplicativo desktop para usar modelos de linguagem localmente no macOS Apple Silicon. A interface SwiftUI controla um serviço Python local, que pode executar modelos via MLX ou encaminhar requisições para Ollama e llama.cpp. O serviço também oferece busca/download no Hugging Face e uma API compatível com OpenAI para IDEs.

## Requisitos

- macOS 13 ou posterior para a interface SwiftUI.
- Mac Apple Silicon para executar modelos com MLX.
- Python 3.10 ou posterior.
- Xcode Command Line Tools (inclui Swift Package Manager).
- Ollama instalado e em execução quando selecionar esse motor, ou um servidor `llama.cpp` iniciado separadamente.

O código da interface está em `macos/`; o serviço Python está em `src/local_ai/`.

## Executar em desenvolvimento

No macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[mlx]"
swift run --package-path macos
```

O aplicativo inicia o backend Python em `127.0.0.1:11435` e o encerra quando a janela é fechada. Para escolher um interpretador Python específico, defina `LOCAL_AI_PYTHON` com o caminho do executável. Se o backend estiver instalado fora da árvore do repositório, defina `LOCAL_AI_BACKEND_PATH` para o diretório que contém `local_ai/`.

Para desenvolver ou testar apenas a API em qualquer sistema com Python:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python -m local_ai
```

Os testes do serviço não precisam baixar modelos nem instalar MLX:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Selecionar motores e modelos

- **MLX:** escolha o runtime MLX e baixe um modelo compatível, como um modelo convertido para MLX. Modelos arbitrários do Hugging Face não são necessariamente compatíveis; o modelo baixado fica selecionável na lista de modelos locais.
- **Ollama:** inicie Ollama separadamente, informe o nome de um modelo já instalado (por exemplo, `llama3.2`) e ajuste a URL se necessário. O padrão é `http://127.0.0.1:11434`.
- **llama.cpp:** inicie o `llama-server` separadamente com a API de chat OpenAI habilitada, informe o nome de modelo servido e ajuste a URL. O padrão é `http://127.0.0.1:8080`.

Busca Hugging Face usa a API pública. Para baixar modelos gated, configure `HF_TOKEN` no ambiente do serviço. O aplicativo exibe falhas de pesquisa/download no status da interface.

## API para VS Code e outras IDEs

Configure uma extensão com suporte a endpoint OpenAI personalizado:

- Base URL: `http://127.0.0.1:11435/v1`
- Modelo: o identificador/caminho mostrado no app
- Chave API: não necessária pelo servidor local

Exemplo de configuração da runtime:

```sh
curl -X POST http://127.0.0.1:11435/api/settings \
  -H 'Content-Type: application/json' \
  -d '{"provider":"mlx","model":"/caminho/para/o/modelo-baixado"}'
```

Teste da API de chat:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Olá!"}]}'
```

Endpoints disponíveis: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `GET/POST /api/settings`, `GET /api/models`, `GET /api/huggingface/search?q=...` e `POST /api/huggingface/download`.

O backend aceita chamadas de chat em formato não-streaming e streaming SSE (`"stream": true`). Ele é deliberadamente vinculado a loopback, rejeita endereços de escuta públicos e só aceita URLs de motores que apontem para loopback, para manter as conversas no computador. Isso não substitui controles de segurança do computador: outros processos do mesmo usuário podem acessar a porta local.

## Limitações do MVP

- A interface SwiftUI exige macOS; não há interface nativa no Windows/Linux.
- O runtime MLX exige Apple Silicon e `mlx-lm`; no ambiente Linux x86_64, apenas o serviço/API pode ser validado.
- Ollama e llama.cpp são conectados a servidores existentes; o aplicativo não instala nem inicia esses motores.
- A busca/download não verifica previamente compatibilidade/licença do modelo com cada runtime. Verifique os termos do modelo e selecione artefatos suportados pelo motor.
- A validação de modelos depende do motor: o modelo de MLX é carregado na primeira inferência; os outros motores respondem de acordo com seu servidor configurado.
