"""Inicia VisionAR AI 2.0 y cierra únicamente sus procesos al salir."""
from __future__ import annotations

import argparse
import webbrowser
from http.server import ThreadingHTTPServer

try:
    from runtime_limits import configure_runtime
except ModuleNotFoundError:
    from scripts.runtime_limits import configure_runtime

configure_runtime()

try:
    from vision_web_server import make_handler
    from vlm_engine import MODEL_SPECS, VLMEngine
except ModuleNotFoundError:
    from scripts.vision_web_server import make_handler
    from scripts.vlm_engine import MODEL_SPECS, VLMEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--llama-port", type=int, default=8091)
    parser.add_argument("--model", choices=MODEL_SPECS, default="smolvlm-500m")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    engine = VLMEngine(port=args.llama_port, selected=args.model)
    print(f"Cargando {MODEL_SPECS[args.model]['name']} en CPU...")
    try:
        engine.start(wait_seconds=90)
    except Exception as exc:
        print(f'VLM no disponible: {exc}. El dashboard y YOLO siguen disponibles.')
    if engine.external:
        print("Reutilizando llama-server externo; VisionAR no lo cerrará ni cambiará su modelo")

    address = ("127.0.0.1", args.web_port)
    server = ThreadingHTTPServer(address, make_handler(engine.url, model_manager=engine))
    url = f"http://{address[0]}:{address[1]}"
    print(f"VisionAR AI 2.0 listo en {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Cerrando VisionAR AI...")
    finally:
        server.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
