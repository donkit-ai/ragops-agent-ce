"""
MCP Server for managing docker-compose services for RAGOps.

Provides tools to:
- List available services
- Initialize compose files in project
- Start/stop services
- Check service status
- Get logs
"""

import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)
import base64
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Self

from donkit_ragops.schemas.config_schemas import RagConfig
from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator

# Package root (where compose files are stored)
PACKAGE_ROOT = Path(__file__).parent.parent.parent
COMPOSE_DIR = PACKAGE_ROOT / "compose"
SERVICES_DIR = COMPOSE_DIR / "services"
TEMPLATES_DIR = COMPOSE_DIR / "templates"

# Compose file name
COMPOSE_FILE = "docker-compose.yml"

# Available services (using Docker Compose profiles)
AVAILABLE_SERVICES = {
    "qdrant": {
        "name": "qdrant",
        "description": "Qdrant vector database for RAG",
        "profile": "qdrant",
        "ports": ["6333:6333", "6334:6334"],
        "url": "http://localhost:6333",
    },
    "chroma": {
        "name": "chroma",
        "description": "Chroma vector database for RAG",
        "profile": "chroma",
        "ports": ["8015:8000"],
        "url": "http://localhost:8015",
    },
    "milvus": {
        "name": "milvus",
        "description": "Milvus vector database for RAG",
        "profile": "milvus",
        "ports": ["19530:19530", "9091:9091"],
        "url": "http://localhost:19530",
    },
    "rag-service": {
        "name": "rag-service",
        "description": "RAG Query service",
        "profile": "rag-service",
        "ports": ["8000:8000"],
        "url": "http://localhost:8000",
    },
}

RAG_SERVICE_API = """
Endpoints
- POST /api/query/stream – streaming final response
- POST /api/query/search –
        Returns the most relevant document chunks based on the query.
        This route just use retriever without any options. Result may be inaccurate.
- POST /api/query/evaluation – evaluation or not streaming result.
All POST endpoints use body:
{
  "query": "string"
}
"""


def check_docker_installed() -> tuple[bool, str]:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, "Docker is running"
        return False, "Docker is installed but not running"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"


def check_docker_compose_installed() -> tuple[bool, str]:
    """Check if docker-compose is installed."""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "docker-compose command failed"
    except FileNotFoundError:
        # Try 'docker compose' (new syntax)
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
        except Exception:
            pass
        return False, "docker-compose is not installed"
    except Exception as e:
        return False, f"Error checking docker-compose: {str(e)}"


# Cache for WSL2 detection
_is_wsl2_cache: bool | None = None


def is_wsl2_docker() -> bool:
    """Check if running on Windows with Docker in WSL2."""
    global _is_wsl2_cache

    if _is_wsl2_cache is not None:
        return _is_wsl2_cache

    if platform.system() != "Windows":
        _is_wsl2_cache = False
        return False

    # Check if docker command works and is WSL2-based
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.OperatingSystem}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # If Docker reports Linux, it's likely WSL2 on Windows
        _is_wsl2_cache = "linux" in result.stdout.lower()
        return _is_wsl2_cache
    except Exception:
        _is_wsl2_cache = False
        return False


def convert_path_for_docker(path: Path) -> str:
    """Convert Windows path to WSL2 path format if needed."""
    path_str = str(path)

    if not is_wsl2_docker():
        return path_str

    # Convert Windows path to WSL2 format
    # C:\Users\... -> /mnt/c/Users/...
    if len(path_str) > 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    return path_str


def get_compose_command() -> list[str]:
    """Get the appropriate docker-compose command."""
    # Try new syntax first
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return ["docker", "compose"]
    except Exception:
        return ["docker-compose"]


server = FastMCP(
    "ragops-compose-manager",
)


def generate_env_file(
    project_id: str,
    rag_config: RagConfig,
    llm_provider: str | None,
    llm_model: str | None,
    openai_api_key: str | None,
    openai_base_url: str | None,
    openai_embeddings_model: str | None,
    azure_openai_api_key: str | None,
    azure_openai_api_version: str | None,
    azure_openai_endpoint: str | None,
    azure_openai_deployment: str | None,
    azure_openai_embeddings_deployment: str | None,
    vertex_credentials_json: str | None,
    ollama_base_url: str | None,
    ollama_api_key: str | None,
    ollama_chat_model: str | None,
    ollama_embedding_model: str | None,
    donkit_api_key: str | None,
    donkit_base_url: str | None,
    log_level: str | None,
) -> str:
    """Generate .env file content from RagConfig."""
    if not rag_config:
        raise ValueError("Rag_config must be provided to the env generator")
    lines = [
        "# =============================================================================",
        "# RAGOps Agent CE - Docker Compose Environment Variables",
        "# =============================================================================",
        "# Generated automatically by ragops-compose-manager",
        "",
        "# -----------------------------------------------------------------------------",
        "# Project Configuration",
        "# -----------------------------------------------------------------------------",
        "",
        f"PROJECT_ID={project_id}",
        f"QDRANT_CONTAINER_NAME={project_id}_qdrant",
        f"CHROMA_CONTAINER_NAME={project_id}_chroma",
        f"MILVUS_ETCD_CONTAINER_NAME={project_id}_milvus_etcd",
        f"MILVUS_MINIO_CONTAINER_NAME={project_id}_milvus_minio",
        f"MILVUS_STANDALONE_CONTAINER_NAME={project_id}_milvus_standalone",
        f"RAG_SERVICE_CONTAINER_NAME={project_id}_rag_service",
        "",
        "# -----------------------------------------------------------------------------",
        "# LLM Provider Credentials",
        "# -----------------------------------------------------------------------------",
        "",
    ]

    # LLM Provider Selection
    lines.append("# LLM Provider Selection")
    lines.append(f"LLM_PROVIDER={llm_provider or ''}")
    lines.append(f"LLM_MODEL={llm_model or ''}")
    lines.append("")

    # OpenAI
    lines.append("# OpenAI")
    lines.append(f"OPENAI_API_KEY={openai_api_key or ''}")
    lines.append(f"OPENAI_BASE_URL={openai_base_url or 'https://api.openai.com/v1'}")
    lines.append(f"OPENAI_EMBEDDINGS_MODEL={openai_embeddings_model or 'text-embedding-3-small'}")
    lines.append("")

    # Azure OpenAI - названия соответствуют Settings в rag_service/core/settings.py
    lines.append("# Azure OpenAI")
    lines.append(f"AZURE_OPENAI_API_KEY={azure_openai_api_key or ''}")
    lines.append(f"AZURE_OPENAI_AZURE_ENDPOINT={azure_openai_endpoint or ''}")
    lines.append("AZURE_OPENAI_API_VERSION=2024-02-15-preview")
    lines.append(f"AZURE_OPENAI_DEPLOYMENT={azure_openai_deployment or ''}")
    lines.append(f"AZURE_OPENAI_API_VERSION={azure_openai_api_version or ''}")
    lines.append(f"AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT={azure_openai_embeddings_deployment or ''}")
    lines.append("")

    # Donkit
    lines.append("# Donkit")
    lines.append(f"DONKIT_API_KEY={donkit_api_key or ''}")
    lines.append(f"DONKIT_BASE_URL={donkit_base_url or 'https://api.donkit.ai'}")

    # Vertex AI
    lines.append("# Vertex AI (Google Cloud)")
    lines.append("# Pass credentials as base64-encoded JSON")
    if vertex_credentials_json:
        # Encode to base64 to avoid issues with special characters in .env
        encoded = base64.b64encode(vertex_credentials_json.encode("utf-8")).decode("utf-8")
        lines.append(f"RAGOPS_VERTEX_CREDENTIALS_JSON={encoded}")
    else:
        lines.append("RAGOPS_VERTEX_CREDENTIALS_JSON=")
    lines.append("")

    # Ollama
    lines.append("# Ollama (Local LLM)")
    ollama_uri = (
        ollama_base_url.replace("localhost", "host.docker.internal")
        if ollama_base_url
        else "http://host.docker.internal:11434/v1"
    )
    lines.append(f"OLLAMA_BASE_URL={ollama_uri}")
    lines.append(f"OLLAMA_API_KEY={ollama_api_key or 'ollama'}")
    lines.append(f"OLLAMA_CHAT_MODEL={ollama_chat_model or 'mistral'}")
    lines.append(f"OLLAMA_EMBEDDING_MODEL={ollama_embedding_model or 'nomic-embed-text'}")
    lines.append("")

    lines.append("# -----------------------------------------------------------------------------")
    lines.append("# RAG Service Configuration")
    lines.append("# -----------------------------------------------------------------------------")
    lines.append("")

    # Encode to base64 to avoid issues with special characters in .env
    if rag_config:
        config_json = rag_config.model_dump_json()
        encoded_config = base64.b64encode(config_json.encode("utf-8")).decode("utf-8")
        lines.append("# RAG Configuration (auto-generated from RagConfig, base64-encoded)")
        lines.append(f"CONFIG={encoded_config}")
    else:
        lines.append("CONFIG=")

    lines.append("")
    lines.append("# -----------------------------------------------------------------------------")
    lines.append("# Server Settings")
    lines.append("# -----------------------------------------------------------------------------")
    lines.append("")

    level = log_level or "INFO"
    lines.append(f"LOG_LEVEL={level}")
    lines.append("")

    return "\n".join(lines)


class InitProjectComposeArgs(BaseModel):
    project_id: str = Field(description="Project ID")
    rag_config: RagConfig = Field(description="RAG service configuration")

    @model_validator(mode="after")
    def _set_default_collection_name(self) -> Self:
        """Ensure retriever_options.collection_name is set.
        If missing/empty, use project_id as a sensible default.
        For Milvus, ensure collection name starts with underscore or letter.
        """
        if not getattr(self.rag_config.retriever_options, "collection_name", None):
            self.rag_config.retriever_options.collection_name = self.project_id

        # Fix collection name for Milvus if needed
        if self.rag_config.db_type == "milvus":
            collection_name = self.rag_config.retriever_options.collection_name
            if not re.match(r"^[a-zA-Z_]", collection_name):
                self.rag_config.retriever_options.collection_name = f"_{collection_name}"
        return self


class StopContainerArgs(BaseModel):
    container_id: str = Field(description="Container ID or name")


class ServicePort(BaseModel):
    """Custom port mapping for a service."""

    service: Literal["qdrant", "chroma", "milvus", "rag-service"] = Field(
        description="Service name"
    )
    port: str = Field(
        description="Host port mapping in format 'host_port:container_port' (e.g., '6335:6333') "
        "or just host port (e.g., '6335')"
    )


class StartServiceArgs(BaseModel):
    service: Literal["qdrant", "chroma", "milvus", "rag-service"] = Field(
        description="Service name (qdrant, chroma, milvus, rag-service)"
    )
    project_id: str = Field(description="Project ID")
    detach: bool = Field(True, description="Run in detached mode")
    build: bool = Field(False, description="Build images before starting")
    custom_ports: list[ServicePort] | None = Field(
        None,
        description=(
            "Custom port mappings for services. "
            "Example: [{'service': 'qdrant', 'port': '6335:6333'}, "
            "{'service': 'rag-service', 'port': '8001:8000'}]"
        ),
    )


class StopServiceArgs(BaseModel):
    service: str = Field(description="Service name")
    project_id: str = Field(description="Project ID")
    remove_volumes: bool = Field(False, description="Remove volumes")


class ServiceStatusArgs(BaseModel):
    service: str | None = Field(None, description="Service name (optional, default: all)")
    project_id: str = Field(description="Project ID")


class GetLogsArgs(BaseModel):
    service: str = Field(description="Service name")
    tail: int = Field(100, description="Number of lines to show")
    project_id: str = Field(description="Project ID")


@server.tool(
    name="list_available_services",
    description="Get list of available Docker Compose services that can be started",
)
async def list_available_services() -> str:
    """List available services."""
    result = {
        "services": list(AVAILABLE_SERVICES.values()),
        "compose_dir": str(SERVICES_DIR),
    }
    return json.dumps(result, indent=2)


@server.tool(
    name="init_project_compose",
    description="Initialize docker-compose file in the project directory with RAG configuration",
)
async def init_project_compose(args: InitProjectComposeArgs) -> str:
    """Initialize compose files in project."""
    compose_target = Path(f"projects/{args.project_id}").resolve()
    # Create compose directory
    compose_target.mkdir(parents=True, exist_ok=True)

    copied_files = []

    # Copy single docker-compose.yml file
    source = SERVICES_DIR / COMPOSE_FILE
    target = compose_target / COMPOSE_FILE

    if source.exists():
        shutil.copy2(source, target)
        copied_files.append(f"compose/{COMPOSE_FILE}")
    else:
        return json.dumps(
            {"status": "error", "message": f"Source compose file not found: {source}"}
        )

    # Read Vertex credentials if path provided
    vertex_credentials_json = None
    vertex_creds_path = os.getenv("RAGOPS_VERTEX_CREDENTIALS")
    if vertex_creds_path and Path(vertex_creds_path).exists():
        try:
            # Read and minify JSON (remove whitespace)
            creds_data = json.loads(Path(vertex_creds_path).read_text())
            vertex_credentials_json = json.dumps(creds_data, separators=(",", ":"))
        except Exception:
            pass  # If reading fails, will pass None

    # Generate .env file with RAG configuration
    env_content = generate_env_file(
        project_id=args.project_id,
        rag_config=args.rag_config,
        llm_provider=os.getenv("RAGOPS_LLM_PROVIDER"),
        llm_model=os.getenv("RAGOPS_LLM_MODEL"),
        openai_api_key=os.getenv("RAGOPS_OPENAI_API_KEY"),
        openai_base_url=os.getenv("RAGOPS_OPENAI_BASE_URL"),
        openai_embeddings_model=os.getenv("RAGOPS_OPENAI_EMBEDDINGS_MODEL"),
        azure_openai_api_key=os.getenv("RAGOPS_AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("RAGOPS_AZURE_OPENAI_ENDPOINT"),
        azure_openai_deployment=os.getenv("RAGOPS_AZURE_OPENAI_DEPLOYMENT"),
        azure_openai_embeddings_deployment=os.getenv("RAGOPS_AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
        azure_openai_api_version=os.getenv("RAGOPS_AZURE_OPENAI_API_VERSION"),
        vertex_credentials_json=vertex_credentials_json,
        ollama_base_url=os.getenv("RAGOPS_OLLAMA_BASE_URL"),
        ollama_api_key=os.getenv("RAGOPS_OLLAMA_API_KEY"),
        ollama_chat_model=os.getenv("RAGOPS_OLLAMA_CHAT_MODEL"),
        ollama_embedding_model=os.getenv("RAGOPS_OLLAMA_EMBEDDINGS_MODEL"),
        log_level=os.getenv("RAGOPS_LOG_LEVEL"),
        donkit_api_key=os.getenv("RAGOPS_DONKIT_API_KEY"),
        donkit_base_url=os.getenv("RAGOPS_DONKIT_BASE_URL"),
    )
    env_file = compose_target / ".env"
    env_file.write_text(env_content)
    copied_files.append("compose/.env")
    result = {
        "status": "success",
        "copied_files": copied_files,
        "message": f"Compose files initialized in {compose_target}",
        "rag_config_applied": args.rag_config is not None,
    }

    return json.dumps(result, indent=2)


class ListContainersArgs(BaseModel):
    """Empty args model for list_containers (FastMCP requires args parameter)."""

    pass


@server.tool(
    name="list_containers",
    description=(
        "List Docker containers,"
        "if want to analyze whether container from another project occupies the same port"
    ),
)
async def list_containers() -> str:
    """List Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", r"{{json .}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            containers = []
            for line in result.stdout.strip().splitlines():
                try:
                    container_info = json.loads(line)
                    containers.append(container_info)
                except json.JSONDecodeError:
                    continue
            return json.dumps({"status": "success", "containers": containers}, indent=2)
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Failed to list containers",
                    "error": result.stderr,
                }
            )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@server.tool(
    name="stop_container",
    description=(
        "Stop Docker container,"
        "if want to stop container from another project that occupies the same port"
    ),
)
async def stop_container(args: StopContainerArgs) -> str:
    """Stop a Docker container by ID or name."""
    container_id = args.container_id

    try:
        result = subprocess.run(
            ["docker", "stop", container_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.dumps({"status": "success", "message": f"Container {container_id} stopped"})
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to stop container {container_id}",
                    "error": result.stderr,
                }
            )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@server.tool(
    name="start_service",
    description=(
        "Start a Docker Compose service,"
        "if want to redeploy with another configuration use init_project_compose first"
    ),
)
async def start_service(args: StartServiceArgs) -> str:
    """Start a service."""
    service = args.service
    project_path = Path(f"projects/{args.project_id}").resolve()
    detach = args.detach
    build = args.build

    # Check Docker
    docker_ok, docker_msg = check_docker_installed()
    if not docker_ok:
        return json.dumps({"status": "error", "message": docker_msg})

    compose_ok, compose_msg = check_docker_compose_installed()
    if not compose_ok:
        return json.dumps({"status": "error", "message": compose_msg})

    # Check service exists
    if service not in AVAILABLE_SERVICES:
        return json.dumps(
            {
                "status": "error",
                "message": f"Unknown service: {service}. "
                f"Available: {list(AVAILABLE_SERVICES.keys())}",
            }
        )

    # Get compose file and profile
    compose_file = project_path / COMPOSE_FILE
    profile = AVAILABLE_SERVICES[service]["profile"]

    if not compose_file.exists():
        return json.dumps(
            {
                "status": "error",
                "message": f"Compose file not found: {compose_file}. "
                f"Run init_project_compose first.",
            }
        )

    # Build command with profile and explicit project name
    cmd = get_compose_command()
    cmd.extend(
        [
            "-f",
            convert_path_for_docker(compose_file),
            "--project-name",
            f"ragops-{args.project_id}",
            "--profile",
            profile,
            "up",
        ]
    )

    if detach:
        cmd.append("-d")
    if build:
        cmd.append("--build")

    # Set custom ports via environment variables if provided
    env = os.environ.copy()
    if args.custom_ports:
        # Create a mapping for easy lookup
        port_map = {sp.service: sp.port for sp in args.custom_ports}

        # For qdrant: QDRANT_PORT_HTTP and QDRANT_PORT_GRPC
        # For chroma: CHROMA_PORT
        # For milvus: MILVUS_PORT and MILVUS_METRICS_PORT
        # For rag-service: RAG_SERVICE_PORT
        if service == "qdrant" and "qdrant" in port_map:
            port_mapping = port_map["qdrant"]
            if ":" in port_mapping:
                host_port = port_mapping.split(":")[0]
                env["QDRANT_PORT_HTTP"] = host_port
                # Assume GRPC port is HTTP port + 1
                env["QDRANT_PORT_GRPC"] = str(int(host_port) + 1)
        elif service == "chroma" and "chroma" in port_map:
            port_mapping = port_map["chroma"]
            if ":" in port_mapping:
                host_port = port_mapping.split(":")[0]
                env["CHROMA_PORT"] = host_port
        elif service == "milvus" and "milvus" in port_map:
            port_mapping = port_map["milvus"]
            if ":" in port_mapping:
                host_port = port_mapping.split(":")[0]
                env["MILVUS_PORT"] = host_port
                # Assume metrics port is main port + 1
                env["MILVUS_METRICS_PORT"] = str(int(host_port) + 1)
        elif service == "rag-service" and "rag-service" in port_map:
            port_mapping = port_map["rag-service"]
            if ":" in port_mapping:
                host_port = port_mapping.split(":")[0]
                env["RAG_SERVICE_PORT"] = host_port

    try:
        # Don't use cwd on Windows with WSL2 Docker - paths are already absolute
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 120,
            "env": env,
        }
        if not is_wsl2_docker():
            run_kwargs["cwd"] = project_path

        result = subprocess.run(cmd, **run_kwargs)

        if result.returncode == 0:
            service_info = AVAILABLE_SERVICES[service]
            # Use custom ports if provided, otherwise use default
            ports = service_info["ports"]
            url = service_info.get("url")

            if args.custom_ports:
                # Create a mapping for easy lookup
                port_map = {sp.service: sp.port for sp in args.custom_ports}

                if service == "qdrant" and "qdrant" in port_map:
                    port_mapping = port_map["qdrant"]
                    host_port = port_mapping.split(":")[0] if ":" in port_mapping else port_mapping
                    ports = [port_mapping, f"{int(host_port) + 1}:6334"]
                    url = f"http://localhost:{host_port}"
                elif service == "chroma" and "chroma" in port_map:
                    port_mapping = port_map["chroma"]
                    host_port = port_mapping.split(":")[0] if ":" in port_mapping else port_mapping
                    ports = [port_mapping]
                    url = f"http://localhost:{host_port}"
                elif service == "milvus" and "milvus" in port_map:
                    port_mapping = port_map["milvus"]
                    host_port = port_mapping.split(":")[0] if ":" in port_mapping else port_mapping
                    ports = [port_mapping, f"{int(host_port) + 1}:9091"]
                    url = f"http://localhost:{host_port}"
                elif service == "rag-service" and "rag-service" in port_map:
                    port_mapping = port_map["rag-service"]
                    host_port = port_mapping.split(":")[0] if ":" in port_mapping else port_mapping
                    ports = [port_mapping]
                    url = f"http://localhost:{host_port}"
            success_result = {
                "status": "success",
                "service": service,
                "message": f"{service_info['description']} started successfully",
                "url": url,
                "ports": ports,
                "custom_ports_applied": args.custom_ports is not None,
                "output": result.stdout,
            }
            if service == "rag-service":
                success_result["rag_service_api_reference"] = RAG_SERVICE_API
            return json.dumps(
                success_result,
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Failed to start service",
                    "error": result.stderr,
                }
            )

    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "message": "Command timed out after 120 seconds"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@server.tool(
    name="stop_service",
    description="Stop a Docker Compose service",
)
async def stop_service(args: StopServiceArgs) -> str:
    """Stop a service."""
    service = args.service
    project_path = Path(f"projects/{args.project_id}").resolve()
    remove_volumes = args.remove_volumes

    if service not in AVAILABLE_SERVICES:
        return json.dumps({"status": "error", "message": f"Unknown service: {service}"})

    compose_file = project_path / COMPOSE_FILE
    profile = AVAILABLE_SERVICES[service]["profile"]

    if not compose_file.exists():
        return json.dumps({"status": "error", "message": f"Compose file not found: {compose_file}"})

    cmd = get_compose_command()
    cmd.extend(
        [
            "-f",
            convert_path_for_docker(compose_file),
            "--project-name",
            f"ragops-{args.project_id}",
            "--profile",
            profile,
            "down",
        ]
    )

    if remove_volumes:
        cmd.append("-v")

    try:
        # Don't use cwd on Windows with WSL2 Docker - paths are already absolute
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 60,
        }
        if not is_wsl2_docker():
            run_kwargs["cwd"] = project_path

        result = subprocess.run(cmd, **run_kwargs)
        if result.returncode == 0:
            return json.dumps(
                {
                    "status": "success",
                    "service": service,
                    "message": f"{service} stopped successfully",
                    "output": result.stdout,
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"status": "error", "message": "Failed to stop service", "error": result.stderr}
            )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@server.tool(
    name="service_status",
    description="Check status of Docker Compose services",
)
async def service_status(args: ServiceStatusArgs) -> str:
    """Check service status."""
    service = args.service
    project_path = Path(f"projects/{args.project_id}").resolve()

    if not project_path.exists():
        return json.dumps({"status": "error", "message": "Project directory not found"})

    # Get list of services to check
    services_to_check = [service] if service else list(AVAILABLE_SERVICES.keys())

    compose_file = project_path / COMPOSE_FILE

    if not compose_file.exists():
        return json.dumps({"status": "error", "message": f"Compose file not found: {compose_file}"})

    statuses = []
    cmd = get_compose_command()

    for svc in services_to_check:
        if svc not in AVAILABLE_SERVICES:
            continue

        profile = AVAILABLE_SERVICES[svc]["profile"]

        try:
            # Don't use cwd on Windows with WSL2 Docker
            run_kwargs = {
                "capture_output": True,
                "text": True,
                "timeout": 10,
            }
            if not is_wsl2_docker():
                run_kwargs["cwd"] = project_path

            result = subprocess.run(
                [
                    *cmd,
                    "-f",
                    convert_path_for_docker(compose_file),
                    "--project-name",
                    f"ragops-{args.project_id}",
                    "--profile",
                    profile,
                    "ps",
                    "--format",
                    "json",
                ],
                **run_kwargs,
            )

            if result.returncode == 0 and result.stdout.strip():
                # docker-compose ps --format json can return:
                # - Array: [{"Name": ...}, ...]
                # - Single object: {"Name": ...}
                # - NDJSON (newline delimited): {"Name": ...}\n{"Name": ...}
                stdout = result.stdout.strip()

                if stdout.startswith("["):
                    # Array format
                    containers = json.loads(stdout)
                elif "\n" in stdout:
                    # NDJSON format - multiple JSON objects separated by newlines
                    containers = [json.loads(line) for line in stdout.split("\n") if line.strip()]
                else:
                    # Single JSON object
                    containers = [json.loads(stdout)]

                statuses.append(
                    {
                        "service": svc,
                        "status": "running" if containers else "stopped",
                        "containers": containers,
                    }
                )
            else:
                statuses.append({"service": svc, "status": "stopped", "containers": []})

        except Exception as e:
            statuses.append({"service": svc, "status": "error", "error": str(e)})

    return json.dumps({"services": statuses}, indent=2)


@server.tool(
    name="get_logs",
    description="Get logs from a Docker Compose service",
)
async def get_logs(args: GetLogsArgs) -> str:
    """Get service logs."""
    service = args.service
    tail = args.tail
    project_path = Path(f"projects/{args.project_id}").resolve()

    if service not in AVAILABLE_SERVICES:
        return json.dumps({"status": "error", "message": f"Unknown service: {service}"})

    compose_file = project_path / COMPOSE_FILE
    profile = AVAILABLE_SERVICES[service]["profile"]

    if not compose_file.exists():
        return json.dumps({"status": "error", "message": f"Compose file not found: {compose_file}"})

    cmd = get_compose_command()
    cmd.extend(
        [
            "-f",
            convert_path_for_docker(compose_file),
            "--project-name",
            f"ragops-{args.project_id}",
            "--profile",
            profile,
            "logs",
            "--tail",
            str(tail),
            service,  # Add service name to get logs for specific service
        ]
    )

    try:
        # Don't use cwd on Windows with WSL2 Docker - paths are already absolute
        run_kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        if not is_wsl2_docker():
            run_kwargs["cwd"] = project_path

        result = subprocess.run(cmd, **run_kwargs)

        return json.dumps(
            {
                "service": service,
                "logs": result.stdout,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
