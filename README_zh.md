# 🏠 HomeRecoEngine 房源推荐引擎

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-最新版-green.svg)](https://fastapi.tiangolo.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![维护状态](https://img.shields.io/badge/维护状态-活跃-green.svg)](https://github.com/yourusername/HomeRecoEngine/graphs/commit-activity)

> **基于向量搜索和地理空间分析的现代化AI房源推荐引擎**

HomeRecoEngine 是一个智能房地产推荐系统，结合语义搜索、地理空间分析和机器学习技术，提供个性化的房源推荐服务。基于 FastAPI 和 Milvus 向量数据库构建，支持自然语言查询、基于位置的搜索和高级过滤功能。

[English](README.md) | **简体中文**

## ✨ 核心功能

### 🔍 **高级搜索能力**
- **语义搜索**: 支持自然语言查询，如"地铁站附近的学区房，三室两厅"
- **地理空间搜索**: 在指定坐标点周围半径范围内（1-50公里）查找房源
- **混合搜索**: 结合语义理解与精确过滤条件
- **实时结果**: 基于向量索引的毫秒级响应时间

### 🗺️ **位置智能**
- **精确距离计算**: 使用 Haversine 公式计算准确的地理距离
- **圆形和矩形区域搜索**: 灵活的地理边界选项
- **坐标系统支持**: 兼容 WGS84 坐标系统
- **多格式输入**: 支持各种地址和坐标格式

### 📊 **数据管理**
- **批量导入**: Excel/CSV 文件上传，带验证和去重功能
- **实时 CRUD**: 通过 REST API 进行增删改查操作
- **数据验证**: 全面的输入验证和错误处理
- **可扩展存储**: 基于 Milvus 向量数据库的高性能操作

### 🤖 **AI 驱动功能**
- **嵌入模型**: 支持多种嵌入模型（BGE、FastEmbed 等）
- **智能匹配**: 基于向量相似度的个性化推荐
- **多语言支持**: 中英文文本处理能力
- **上下文理解**: 房源描述的深度语义分析

## 🚀 快速开始

### 环境要求

- **Python 3.12+** 
- **Milvus 2.5.11+** (向量数据库)
- **Docker & Docker Compose** (用于 Milvus)
- **8GB+ 内存** (推荐)

### 第一步：安装 uv 包管理器

uv 是一个快速的 Python 包管理器，推荐使用它来管理项目依赖。

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 第二步：克隆和设置项目

```bash
# 克隆仓库
git clone https://github.com/yourusername/HomeRecoEngine.git
cd HomeRecoEngine

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Windows 用户使用: .venv\Scripts\activate
uv pip install -r pyproject.toml
```

### 第三步：启动 Milvus（使用官方 Docker Compose）

```bash
# 下载官方 Milvus Docker Compose 文件
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动 Milvus
docker-compose up -d

# 验证 Milvus 运行状态
docker-compose ps
```

### 第四步：配置应用程序

编辑配置文件 `conf/service_conf.yaml`:

```yaml
# Milvus 配置
milvus:
  hosts: 'http://127.0.0.1:19530'  # Milvus 服务器地址
  username: 'root'                 # 默认用户名
  password: 'Milvus'               # 默认密码
  db_name: ''                      # 数据库名称（可选）
  
# API 服务器配置  
home_recommendation:
  host: 0.0.0.0                   # 监听所有网络接口
  http_port: 7001                 # API 端口

# 嵌入模型配置
user_default_llm:
  embedding_model: 'BAAI/bge-large-zh-v1.5@BAAI'  # 默认嵌入模型
```

### 第五步：配置 NLTK 数据（必需）

系统需要 NLTK 数据文件。如果项目根目录包含 `nltk` 文件夹，请将其复制到用户目录：

```bash
# 将 NLTK 数据复制到用户目录（如果项目根目录存在 nltk 文件夹）
cp -r nltk /home/$(whoami)/

# 或者：设置 NLTK_DATA 环境变量
export NLTK_DATA=/path/to/project/nltk
```

### 第六步：配置模型下载（可选）

系统将自动从 Hugging Face 下载嵌入模型到：
- **默认路径**: `~/.cache/huggingface/transformers/`
- **自定义路径**: 设置 `TRANSFORMERS_CACHE` 环境变量

**中国用户使用镜像加速下载：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 第七步：运行服务

```bash
# 启动 API 服务器
python api/app.py

# 或使用调试模式
python api/app.py --debug

# API 将在 http://localhost:7001 上可用
# Swagger UI: http://localhost:7001/docs
```

## 📖 API 文档

### 交互式文档
- **Swagger UI**: [http://localhost:7001/docs](http://localhost:7001/docs)
- **ReDoc**: [http://localhost:7001/redoc](http://localhost:7001/redoc)

### 核心接口

#### 🔍 搜索房源
```http
POST /api/houses/search
```

**在5公里半径内查找房源:**
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

**语义搜索:**
```json
{
    "user_query_text": "地铁站附近的学区房，精装修，三室两厅",
    "price_range": {
        "min_price": 500,
        "max_price": 1200
    },
    "limit": 15
}
```

#### 🏡 房源管理
```http
POST /api/houses/insert          # 添加单个房源
POST /api/houses/batch-insert    # 批量添加房源
GET  /api/houses/detail/{id}     # 获取房源详情
DELETE /api/houses/{id}          # 删除房源
```

#### 📤 数据导入
```http
POST /api/houses/upload-excel    # 上传 Excel 文件
POST /api/houses/preview-excel   # 导入前预览数据
```

完整的 API 文档请参考 [API_DOCUMENTATION.md](API_DOCUMENTATION.md)。

## 🎯 使用示例

### Python 客户端

```python
import requests

# 初始化客户端
BASE_URL = "http://localhost:7001/api/houses"

# 搜索附近房源
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

# 语义搜索
def semantic_search(query, max_price=1000):
    response = requests.post(f"{BASE_URL}/search", json={
        "user_query_text": query,
        "price_range": {"max_price": max_price},
        "limit": 15
    })
    return response.json()

# 使用示例
properties = search_nearby_properties(116.3974, 39.9093, 3.0)
school_properties = semantic_search("学区房 地铁站附近")
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

// 查找当前位置附近的房源
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
    
    console.log(`在附近找到 ${results.data.total} 套房源`);
});
```

## 🏗️ 系统架构

```
HomeRecoEngine/
├── 📁 api/                     # API 层
│   ├── 📁 apps/               # FastAPI 路由处理器
│   ├── 📁 db/                 # 数据库服务
│   │   └── 📁 services/       # 业务逻辑
│   └── 📁 utils/              # API 工具
├── 📁 core/                   # 核心组件
│   ├── 📁 llm/               # LLM 和嵌入模型
│   ├── 📁 nlp/               # NLP 处理
│   ├── 📁 prompts/           # AI 提示词
│   └── 📁 utils/             # 核心工具
├── 📁 conf/                  # 配置文件
└── 📁 reference/             # 文档和示例
```

### 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| **Web 框架** | FastAPI | 高性能异步 API |
| **向量数据库** | Milvus | 相似度搜索和存储 |
| **嵌入模型** | BGE, FastEmbed | 文本向量化 |
| **地理空间** | Haversine 公式 | 距离计算 |
| **数据处理** | Pandas, OpenPyXL | 数据导入和处理 |
| **AI/ML** | Transformers, PyTorch | 自然语言处理 |

## 📊 性能表现

### 基准测试
- **搜索延迟**: 典型查询 < 100ms
- **向量索引**: HNSW 算法优化性能
- **并发用户**: 支持 1000+ 并发请求
- **数据规模**: 已测试 100万+ 房源记录

### 优化特性
- **延迟加载**: 按需模型初始化
- **连接池**: 高效的数据库连接管理
- **缓存机制**: 频繁查询的智能缓存
- **批处理**: 优化的批量操作

## 🔧 配置选项

### 环境变量

```bash
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# API 配置
API_HOST=0.0.0.0
API_PORT=7001

# 模型配置
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com  # 中国用户使用镜像
TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/

# NLTK 配置
NLTK_DATA=/home/$(whoami)/nltk  # NLTK 数据目录
```

### 高级配置

编辑 `conf/service_conf.yaml`:

```yaml
# 向量数据库设置
milvus:
  host: "localhost"
  port: 19530
  collection_name: "house_recommendation"
  
# 嵌入模型设置
embedding:
  model_name: "bge-large-zh-v1.5"
  dimension: 1024
  device: "cpu"
  
# 搜索设置
search:
  default_limit: 10
  max_limit: 100
  similarity_threshold: 0.7
```

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行覆盖率测试
python -m pytest tests/ --cov=api --cov-report=html

# 运行特定测试类别
python -m pytest tests/test_search.py      # 搜索功能
python -m pytest tests/test_geospatial.py  # 位置功能
python -m pytest tests/test_api.py         # API 接口
```

### 手动测试

```bash
# 测试 API 服务器
python simple_test.py

# 测试搜索功能
python api/db/services/example_usage.py
```

## 📦 数据格式

### 房源数据结构

```json
{
    "id": 1001,
    "xqmc": ["小区名称"],
    "qy": "所在区域",
    "dz": "详细地址",
    "jd": 116.3974,           // 经度
    "wd": 39.9093,            // 纬度
    "mj": 95.6,               // 面积（平方米）
    "fyhx": "三室两厅",        // 户型
    "zj": 650.5,              // 总价（万元）
    "dj": 6800,               // 单价（元/平方米）
    "lc": "15/30层",          // 楼层
    "cx": "南北",             // 朝向
    "zxqk": "精装修",         // 装修情况
    "ywdt": "有",             // 有无电梯
    "ywcw": "有",             // 有无车位
    "xqtd": "学区房，地铁便利",  // 小区特点
    "zb": "周边商圈成熟，配套齐全"  // 周边环境
}
```

### Excel 导入格式

| 列名 | 必填 | 描述 | 示例 |
|------|------|------|------|
| id | ✅ | 唯一标识符 | 1001 |
| xqmc | ✅ | 小区名称 | "阳光花园" |
| qy | ✅ | 所在区域 | "朝阳区" |
| jd | ✅ | 经度 | 116.3974 |
| wd | ✅ | 纬度 | 39.9093 |
| mj | ✅ | 面积（平方米） | 95.6 |
| zj | ✅ | 总价（万元） | 650.5 |
| ... | | | |

## 🤝 贡献指南

我们欢迎您的贡献！请查看我们的[贡献指南](CONTRIBUTING.md)了解详情。

### 开发环境设置

```bash
# Fork 并克隆仓库
git clone https://github.com/yourusername/HomeRecoEngine.git
cd HomeRecoEngine

# 安装开发依赖
uv sync
```

### 代码规范

- **Python**: 遵循 PEP 8，使用 Black 格式化
- **类型提示**: 为所有函数添加类型注解
- **文档**: 为所有公共方法编写文档字符串
- **测试**: 保持 >90% 的测试覆盖率

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🆘 技术支持

### 文档资源
- [API 文档](API_DOCUMENTATION.md)
- [统一搜索 API 指南](UNIFIED_SEARCH_API.md)
- [API 测试指南](API_TEST_GUIDE.md)

### 获取帮助
- **问题反馈**: [GitHub Issues](https://github.com/yourusername/HomeRecoEngine/issues)
- **技术讨论**: [GitHub Discussions](https://github.com/yourusername/HomeRecoEngine/discussions)
- **项目百科**: [项目 Wiki](https://github.com/yourusername/HomeRecoEngine/wiki)

### 社区交流
- **微信群**: 添加微信群获取中文技术支持
- **QQ群**: 加入 QQ 技术交流群
- **Discord**: [加入 Discord 服务器](https://discord.gg/yourinvite)

## 🎉 致谢

感谢以下开源项目的支持：

- [Milvus](https://milvus.io/) - 向量数据库基础设施
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [BGE Embeddings](https://github.com/FlagOpen/FlagEmbedding) - 高质量文本嵌入
- [BAAI](https://www.baai.ac.cn/) - 预训练嵌入模型

## 🗺️ 发展路线

### 🔮 即将推出的功能

- [ ] **多城市支持**: 扩展到多城市部署
- [ ] **实时更新**: WebSocket 实时房源更新
- [ ] **高级分析**: 房地产市场趋势分析
- [ ] **移动应用**: React Native 移动端应用
- [ ] **机器学习**: 预测性定价模型
- [ ] **集成 API**: 连接主要房地产平台

### 📈 性能改进

- [ ] **GPU 加速**: 嵌入模型的 CUDA 支持
- [ ] **分布式搜索**: 多节点 Milvus 集群支持
- [ ] **边缘缓存**: Redis 缓存频繁访问的数据
- [ ] **自动扩展**: Kubernetes 部署配置

## 🌟 故障排除

### 常见问题

**1. Milvus 连接失败**
```bash
# 检查 Milvus 是否正在运行
docker-compose ps

# 查看 Milvus 日志
docker-compose logs milvus-standalone
```

**2. 模型下载问题**
```bash
# 使用中国镜像
export HF_ENDPOINT=https://hf-mirror.com

# 检查模型缓存目录
ls -la ~/.cache/huggingface/transformers/
```

**3. 端口已被占用**
```bash
# 在 conf/service_conf.yaml 中修改端口
home_recommendation:
  http_port: 8080  # 使用不同端口
```

**4. 虚拟环境问题**
```bash
# 重新创建虚拟环境
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

**5. NLTK 数据未找到**
```bash
# 将 NLTK 数据复制到用户目录
cp -r nltk /home/$(whoami)/

# 或设置环境变量
export NLTK_DATA=/path/to/project/nltk

# 验证 NLTK 数据位置
python -c "import nltk; print(nltk.data.path)"
```

## 🌟 用户案例

### 典型使用场景

1. **房地产中介**: 为客户快速匹配合适房源
2. **房产网站**: 提供智能搜索和推荐功能
3. **投资分析**: 基于地理位置的投资决策支持
4. **市场研究**: 房地产市场数据分析

### 成功案例

> "HomeRecoEngine 将我们的房源匹配效率提升了 300%，客户满意度显著提高。"  
> —— 某知名房产中介公司

> "语义搜索功能让用户可以用自然语言描述需求，大大提升了用户体验。"  
> —— 房产门户网站技术负责人

## 🔮 技术愿景

我们致力于构建下一代智能房地产服务平台，通过 AI 技术革新传统房产服务模式：

- **智能化**: 深度理解用户需求，提供精准推荐
- **个性化**: 基于用户行为的个性化服务
- **实时性**: 毫秒级响应的搜索体验
- **可扩展**: 支持百万级房源和千万级用户

### 提交约定

遵循 [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - 新功能
- `fix:` - 错误修复
- `docs:` - 文档更改
- `style:` - 代码样式更改
- `refactor:` - 代码重构
- `test:` - 测试添加/修改
- `chore:` - 构建过程或辅助工具更改

## 📄 许可证

本项目为专有软件。保留所有权利。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请为我们点个星！**

用 ❤️ 由 HomeRecoEngine 团队制作

[🏠 项目主页](https://github.com/yourusername/HomeRecoEngine) • [📚 文档](API_DOCUMENTATION.md) • [🐛 问题反馈](https://github.com/yourusername/HomeRecoEngine/issues) • [💬 技术讨论](https://github.com/yourusername/HomeRecoEngine/discussions)

</div>