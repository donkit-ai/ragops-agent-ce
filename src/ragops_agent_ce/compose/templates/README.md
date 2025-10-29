# Docker Compose Services для RagOps

Эта директория содержит docker-compose файлы для быстрого развёртывания сервисов RagOps.

## 📦 Доступные сервисы

### 1. Qdrant (qdrant.yml)
Векторная база данных для хранения embeddings.

**Порты:**
- `6333` - HTTP API
- `6334` - gRPC API

**Dashboard:** http://localhost:6333/dashboard

### 2. RAG Service (rag-service.yml)
Основной RAG сервис для выполнения запросов к векторной базе.

**Порты:**
- `8000` - HTTP API

**API Docs:** http://localhost:8000/api/docs

### 3. Full Stack (full-stack.yml)
Все сервисы вместе (Qdrant + RAG Service).

## 🚀 Быстрый старт

### Шаг 1: Настройка credentials

Скопируйте `.env.example` в `.env` и заполните нужные credentials:

```bash
cp .env.example .env
nano .env
```

Минимально нужно настроить один LLM провайдер (OpenAI, Azure OpenAI, или Vertex AI).

### Шаг 2: Запуск сервисов

#### Вариант A: Только Qdrant
```bash
docker-compose -f qdrant.yml up -d
```

#### Вариант B: Полный стек
```bash
docker-compose -f full-stack.yml up -d
```

#### Вариант C: Конкретный сервис
```bash
# RAG Service
docker-compose -f rag-service.yml up -d
```

### Шаг 3: Проверка статуса

```bash
docker-compose -f full-stack.yml ps
```

### Шаг 4: Просмотр логов

```bash
# Все сервисы
docker-compose -f full-stack.yml logs -f

# Конкретный сервис
docker-compose -f full-stack.yml logs -f qdrant
docker-compose -f full-stack.yml logs -f rag-service
```

## 🛠️ Управление сервисами

### Остановка
```bash
docker-compose -f full-stack.yml down
```

### Остановка с удалением volumes
```bash
docker-compose -f full-stack.yml down -v
```

### Перезапуск
```bash
docker-compose -f full-stack.yml restart
```

### Обновление образов
```bash
docker-compose -f full-stack.yml pull
docker-compose -f full-stack.yml up -d
```

## 🔧 Настройка через .env

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

**Важно:** Положите JSON файл с credentials в эту же директорию.

## 📊 Проверка работоспособности

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
Если порт уже занят, измените маппинг в compose файле:
```yaml
ports:
  - "6334:6333"  # внешний:внутренний
```

### "Cannot connect to Docker daemon"
Убедитесь что Docker запущен:
```bash
docker info
```

### "Permission denied" для Vertex AI credentials
```bash
chmod 600 vertex_service_account.json
```

### Qdrant не стартует
Проверьте что папка для volume доступна:
```bash
docker volume ls
docker volume inspect qdrant_data
```

## 📝 Примеры использования

### Создание коллекции в Qdrant
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

### Запрос к RAG Service
```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is RAG?"
  }'
```

## 🔗 Полезные ссылки

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [RagOps Agent CE GitHub](https://github.com/donkit-ai/ragops-agent-ce)

## 💡 Советы

1. **Development:** Используйте отдельные compose файлы для каждого сервиса
2. **Production:** Используйте `full-stack.yml` или настройте kubernetes
3. **Monitoring:** Добавьте `--name` для контейнеров для легкой идентификации
4. **Backups:** Регулярно делайте бэкап `qdrant_data` volume
