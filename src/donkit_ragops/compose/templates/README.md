# Docker Compose Services for RAGOps

This directory contains docker-compose files for quick deployment of RAGOps services.

## 📦 Available Services

### 1. Qdrant (qdrant.yml)
Vector database for storing embeddings.

**Ports:**
- `6333` - HTTP API
- `6334` - gRPC API

**Dashboard:** http://localhost:6333/dashboard

### 2. RAG Service (rag-service.yml)
Main RAG service for querying the vector database.

**Ports:**
- `8000` - HTTP API

**API Docs:** http://localhost:8000/api/docs

### 3. Full Stack (full-stack.yml)
All services together (Qdrant + RAG Service).

## 🚀 Quick Start

### Step 1: Configure Credentials

Copy `.env.example` to `.env` and fill in the required credentials:

```bash
cp .env.example .env
nano .env
```

At minimum, you need to configure one LLM provider (OpenAI, Azure OpenAI, or Vertex AI).

### Step 2: Start Services

#### Option A: Qdrant Only
```bash
docker-compose -f qdrant.yml up -d
```

#### Option B: Full Stack
```bash
docker-compose -f full-stack.yml up -d
```

#### Option C: Specific Service
```bash
# RAG Service
docker-compose -f rag-service.yml up -d
```

### Step 3: Check Status

```bash
docker-compose -f full-stack.yml ps
```

### Step 4: View Logs

```bash
# All services
docker-compose -f full-stack.yml logs -f

# Specific service
docker-compose -f full-stack.yml logs -f qdrant
docker-compose -f full-stack.yml logs -f rag-service
```

## 🛠️ Service Management

### Stop Services
```bash
docker-compose -f full-stack.yml down
```

### Stop with Volume Removal
```bash
docker-compose -f full-stack.yml down -v
```

### Restart
```bash
docker-compose -f full-stack.yml restart
```

### Update Images
```bash
docker-compose -f full-stack.yml pull
docker-compose -f full-stack.yml up -d
```

## 🔧 Configuration via .env

### OpenAI
```env
OPENAI_API_KEY=sk-...
```

### Azure OpenAI
```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Vertex AI
```env
GOOGLE_APPLICATION_CREDENTIALS=./vertex_service_account.json
RAGOPS_VERTEX_CREDENTIALS=./vertex_service_account.json
```

**Important:** Place the JSON credentials file in this directory.

## 📊 Health Checks

### Qdrant
```bash
curl http://localhost:6333/health
```

### RAG Service
```bash
curl http://localhost:8000/health
```

## 🐛 Troubleshooting

### "Port already in use"
If the port is already in use, change the mapping in the compose file:
```yaml
ports:
  - "6334:6333"  # external:internal
```

### "Cannot connect to Docker daemon"
Make sure Docker is running:
```bash
docker info
```

### "Permission denied" for Vertex AI credentials
```bash
chmod 600 vertex_service_account.json
```

### Qdrant won't start
Check that the volume folder is accessible:
```bash
docker volume ls
docker volume inspect qdrant_data
```

## 📝 Usage Examples

### Create Collection in Qdrant
```bash
curl -X PUT http://localhost:6333/collections/my_collection \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

### Query RAG Service
```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is RAG?"
  }'
```

## 🔗 Useful Links

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [RAGOps Agent CE GitHub](https://github.com/donkit-ai/donkit-ragops)

## 💡 Tips

1. **Development:** Use separate compose files for each service
2. **Production:** Use `full-stack.yml` or configure Kubernetes
3. **Monitoring:** Add `--name` to containers for easy identification
4. **Backups:** Regularly backup the `qdrant_data` volume
