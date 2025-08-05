# 🏠 HomeRecoEngine

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5.11+-orange.svg)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/HomeRecoEngine/graphs/commit-activity)

> **A modern, AI-powered real estate recommendation engine built with vector search and geospatial capabilities.**

HomeRecoEngine is an intelligent real estate recommendation system that combines semantic search, geospatial analysis, and machine learning to provide personalized property recommendations. Built with FastAPI and Milvus vector database, it offers powerful search capabilities including natural language queries, location-based searches, and advanced filtering.

[English](README.md) | [简体中文](README_zh.md)

## ✨ Features

### 🔍 **Advanced Search Capabilities**
- **Semantic Search**: Natural language queries like "spacious apartment near subway with good schools"
- **Geospatial Search**: Find properties within specific radius (1-50km) from any location
- **Hybrid Search**: Combine semantic understanding with precise filtering
- **Real-time Results**: Sub-second response times with vector indexing

### 🗺️ **Location Intelligence**
- **Precise Distance Calculation**: Haversine formula for accurate geographic distances
- **Circular & Rectangular Area Search**: Flexible geographic boundary options
- **Coordinate Support**: WGS84 coordinate system compatibility
- **Multi-format Input**: Support for various address and coordinate formats

### 📊 **Data Management**
- **Bulk Import**: Excel/CSV file upload with validation and deduplication
- **Real-time CRUD**: Create, read, update, delete operations via REST API
- **Data Validation**: Comprehensive input validation and error handling
- **Scalable Storage**: Milvus vector database for high-performance operations

### 🤖 **AI-Powered Features**
- **Embedding Models**: Support for multiple embedding models (BGE, FastEmbed, etc.)
- **Intelligent Matching**: Vector similarity matching for personalized recommendations
- **Multi-language Support**: Chinese and English text processing
- **Contextual Understanding**: Deep semantic analysis of property descriptions

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** 
- **Milvus 2.5.11+** (Vector Database)
- **Docker & Docker Compose** (For Milvus)
- **8GB+ RAM** (Recommended)

### Step 1: Install uv Package Manager

uv is a fast Python package manager, recommended for managing project dependencies.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/yourusername/HomeRecoEngine.git
cd HomeRecoEngine

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### Step 3: Start Milvus (Using Official Docker Compose)

```bash
# Download official Milvus Docker Compose file
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker-compose up -d

# Verify Milvus is running
docker-compose ps
```

### Step 4: Configure the Application

Edit the configuration file `conf/service_conf.yaml`:

```yaml
# Milvus Configuration
milvus:
  hosts: 'http://127.0.0.1:19530'  # Milvus server address
  username: 'root'                 # Default username
  password: 'Milvus'               # Default password
  db_name: ''                      # Database name (optional)
  
# API Server Configuration  
home_recommendation:
  host: 0.0.0.0                   # Listen on all interfaces
  http_port: 7001                 # API port

# Embedding Model Configuration
user_default_llm:
  embedding_model: 'BAAI/bge-large-zh-v1.5@BAAI'  # Default embedding model
```

### Step 5: Configure NLTK Data (Required)

The system requires NLTK data files. If the project includes a `nltk` folder, copy it to your user directory:

```bash
# Copy NLTK data to user directory (if nltk folder exists in project root)
cp -r nltk /home/$(whoami)/

# Alternative: Set NLTK_DATA environment variable
export NLTK_DATA=/path/to/project/nltk
```

### Step 6: Configure Model Download (Optional)

The system will automatically download embedding models from Hugging Face to:
- **Default path**: `~/.cache/huggingface/transformers/`
- **Custom path**: Set `TRANSFORMERS_CACHE` environment variable

**For users in China, use mirror to speed up downloads:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Step 7: Run the Service

```bash
# Start the API server
uv run python -m api.app

# Or with debug mode
uv run python -m api.app --debug

# The API will be available at http://localhost:7001
# Swagger UI: http://localhost:7001/docs
```

## 📖 API Documentation

### Interactive Documentation
- **Swagger UI**: [http://localhost:7001/docs](http://localhost:7001/docs)
- **ReDoc**: [http://localhost:7001/redoc](http://localhost:7001/redoc)

### Core Endpoints

#### 🔍 Search Properties
```http
POST /api/houses/search
```

**Find properties within 5km radius:**
```json
{
    "location": {
        "center_longitude": 116.3974,
        "center_latitude": 39.9093,
        "radius_km": 5.0
    },
    "price_range": {
        "min_price": 300,
        "max_price": 800
    },
    "limit": 20
}
```

**Semantic search:**
```json
{
    "user_query_text": "luxury apartment near subway with good schools",
    "price_range": {
        "min_price": 500,
        "max_price": 1200
    },
    "limit": 15
}
```

#### 🏡 Property Management
```http
POST /api/houses/insert          # Add single property
POST /api/houses/batch-insert    # Add multiple properties
GET  /api/houses/detail/{id}     # Get property details
DELETE /api/houses/{id}          # Delete property
```

#### 📤 Data Import
```http
POST /api/houses/upload-excel    # Upload Excel file
POST /api/houses/preview-excel   # Preview data before import
```

For complete API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## 🎯 Usage Examples

### Python Client

```python
import requests

# Initialize client
BASE_URL = "http://localhost:7001/api/houses"

# Search properties near a location
def search_nearby_properties(lng, lat, radius_km=5.0):
    response = requests.post(f"{BASE_URL}/search", json={
        "location": {
            "center_longitude": lng,
            "center_latitude": lat,
            "radius_km": radius_km
        },
        "price_range": {"min_price": 300, "max_price": 800},
        "limit": 20
    })
    return response.json()

# Semantic search
def semantic_search(query, max_price=1000):
    response = requests.post(f"{BASE_URL}/search", json={
        "user_query_text": query,
        "price_range": {"max_price": max_price},
        "limit": 15
    })
    return response.json()

# Example usage
properties = search_nearby_properties(116.3974, 39.9093, 3.0)
school_properties = semantic_search("school district apartment")
```

### JavaScript/TypeScript

```typescript
interface SearchParams {
    location?: {
        center_longitude: number;
        center_latitude: number;
        radius_km: number;
    };
    user_query_text?: string;
    price_range?: {
        min_price?: number;
        max_price?: number;
    };
    limit?: number;
}

async function searchProperties(params: SearchParams) {
    const response = await fetch('/api/houses/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    });
    
    return await response.json();
}

// Find properties near current location
navigator.geolocation.getCurrentPosition(async (position) => {
    const results = await searchProperties({
        location: {
            center_longitude: position.coords.longitude,
            center_latitude: position.coords.latitude,
            radius_km: 5.0
        },
        price_range: { min_price: 300, max_price: 800 },
        limit: 20
    });
    
    console.log(`Found ${results.data.total} properties nearby`);
});
```

## 🏗️ Architecture

```
HomeRecoEngine/
├── 📁 api/                     # API Layer
│   ├── 📁 apps/               # FastAPI route handlers
│   ├── 📁 db/                 # Database services
│   │   └── 📁 services/       # Business logic
│   └── 📁 utils/              # API utilities
├── 📁 core/                   # Core Components
│   ├── 📁 llm/               # LLM and embedding models
│   ├── 📁 nlp/               # NLP processing
│   ├── 📁 prompts/           # AI prompts
│   └── 📁 utils/             # Core utilities
├── 📁 conf/                  # Configuration files
└── 📁 reference/             # Documentation and examples
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | High-performance async API |
| **Vector Database** | Milvus | Similarity search and storage |
| **Embedding Models** | BGE, FastEmbed | Text vectorization |
| **Geospatial** | Haversine Formula | Distance calculations |
| **Data Processing** | Pandas, OpenPyXL | Data import and manipulation |
| **AI/ML** | Transformers, PyTorch | Natural language processing |

## 📊 Performance

### Benchmarks
- **Search Latency**: < 100ms for typical queries
- **Vector Indexing**: HNSW algorithm for optimal performance
- **Concurrent Users**: Supports 1000+ concurrent requests
- **Data Scale**: Tested with 1M+ property records

### Optimization Features
- **Lazy Loading**: On-demand model initialization
- **Connection Pooling**: Efficient database connections
- **Caching**: Intelligent caching for frequent queries
- **Batch Processing**: Optimized bulk operations

## 🔧 Configuration

### Environment Variables

```bash
# Milvus Configuration
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# API Configuration
API_HOST=0.0.0.0
API_PORT=7001

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com  # For users in China
TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/

# NLTK Configuration
NLTK_DATA=/home/$(whoami)/nltk  # NLTK data directory
```

### Advanced Configuration

Edit `conf/service_conf.yaml`:

```yaml
# Vector Database Settings
milvus:
  hosts: "http://127.0.0.1:19530"
  username: "root"
  password: "Milvus"
  
# Embedding Model Settings
user_default_llm:
  embedding_model: "BAAI/bge-large-zh-v1.5@BAAI"
  
# API Server Settings
home_recommendation:
  host: "0.0.0.0"
  http_port: 7001
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=api --cov-report=html

# Run specific test categories
python -m pytest tests/test_search.py      # Search functionality
python -m pytest tests/test_geospatial.py  # Location features
python -m pytest tests/test_api.py         # API endpoints
```

### Manual Testing

```bash
# Test the API server
python simple_test.py

# Test search functionality
python api/db/services/example_usage.py
```

## 📦 Data Format

### Property Data Schema

```json
{
    "id": 1001,
    "xqmc": ["Community Name"],
    "qy": "District",
    "dz": "Full Address",
    "jd": 116.3974,           // Longitude
    "wd": 39.9093,            // Latitude
    "mj": 95.6,               // Area (sqm)
    "fyhx": "3BR2BA",         // Layout
    "zj": 650.5,              // Total Price (10k CNY)
    "dj": 6800,               // Unit Price (CNY/sqm)
    "lc": "15/30F",           // Floor
    "cx": "South-North",      // Orientation
    "zxqk": "Renovated",      // Renovation
    "ywdt": "Yes",            // Elevator
    "ywcw": "Yes",            // Parking
    "xqtd": "School district, near subway",  // Features
    "zb": "Mature commercial area nearby"    // Surroundings
}
```

### Excel Import Format

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| id | ✅ | Unique identifier | 1001 |
| xqmc | ✅ | Community name | "Sunshine Garden" |
| qy | ✅ | District | "Chaoyang" |
| jd | ✅ | Longitude | 116.3974 |
| wd | ✅ | Latitude | 39.9093 |
| mj | ✅ | Area (sqm) | 95.6 |
| zj | ✅ | Total price (10k CNY) | 650.5 |
| ... | | | |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/HomeRecoEngine.git
cd HomeRecoEngine

# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/
```

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Docstrings for all public methods
- **Testing**: Maintain >90% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](API_DOCUMENTATION.md)
- [Unified Search API Guide](UNIFIED_SEARCH_API.md)
- [API Testing Guide](API_TEST_GUIDE.md)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/HomeRecoEngine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/HomeRecoEngine/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/HomeRecoEngine/wiki)

### Community
- **Discord**: [Join our Discord server](https://discord.gg/yourinvite)
- **WeChat**: Add WeChat group for Chinese users

## 🎉 Acknowledgments

- [Milvus](https://milvus.io/) - Vector database infrastructure
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding) - High-quality text embeddings
- [BAAI](https://www.baai.ac.cn/) - Pre-trained embedding models

## 🗺️ Roadmap

### 🔮 Upcoming Features

- [ ] **Multi-city Support**: Expand beyond single city deployments
- [ ] **Real-time Updates**: WebSocket for live property updates
- [ ] **Advanced Analytics**: Property market trend analysis
- [ ] **Mobile App**: React Native mobile application
- [ ] **Machine Learning**: Predictive pricing models
- [ ] **Integration APIs**: Connect with major real estate platforms

### 📈 Performance Improvements

- [ ] **GPU Acceleration**: CUDA support for embedding models
- [ ] **Distributed Search**: Multi-node Milvus cluster support
- [ ] **Edge Caching**: Redis for frequently accessed data
- [ ] **Auto-scaling**: Kubernetes deployment configurations

## 🌟 Troubleshooting

### Common Issues

**1. Milvus Connection Failed**
```bash
# Check if Milvus is running
docker-compose ps

# Check Milvus logs
docker-compose logs milvus-standalone
```

**2. Model Download Issues**
```bash
# Use China mirror
export HF_ENDPOINT=https://hf-mirror.com

# Check model cache directory
ls -la ~/.cache/huggingface/transformers/
```

**3. Port Already in Use**
```bash
# Change port in conf/service_conf.yaml
home_recommendation:
  http_port: 8080  # Use different port
```

**4. NLTK Data Not Found**
```bash
# Copy NLTK data to user directory
cp -r nltk /home/$(whoami)/

# Or set environment variable
export NLTK_DATA=/path/to/project/nltk

# Verify NLTK data location
python -c "import nltk; print(nltk.data.path)"
```

### Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/modifications
- `chore:` - Build process or auxiliary tool changes

## 📄 License

This project is proprietary software. All rights reserved.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the HomeRecoEngine team

[🏠 Homepage](https://github.com/yourusername/HomeRecoEngine) • [📚 Docs](API_DOCUMENTATION.md) • [🐛 Issues](https://github.com/yourusername/HomeRecoEngine/issues) • [💬 Discussions](https://github.com/yourusername/HomeRecoEngine/discussions)

</div>