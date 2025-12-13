#!/bin/bash
# =====================================================
# 网络攻击检测系统 - 一键部署脚本
# Network Attack Detection System - Quick Start
# =====================================================

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║       网络攻击检测系统 - 一键部署                     ║"
echo "║   基于多源数据融合的深度学习网络攻击检测              ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: Docker未安装${NC}"
        echo "请先安装Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}错误: Docker Compose未安装${NC}"
        echo "请先安装Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi

    echo -e "${GREEN}✓ Docker环境检测通过${NC}"
}

# 创建必要的目录
create_directories() {
    echo "创建数据目录..."
    mkdir -p data/raw data/processed data/logs
    mkdir -p outputs/checkpoints outputs/results outputs/figures outputs/reports outputs/logs
    mkdir -p notebooks
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 构建Docker镜像
build_image() {
    echo ""
    echo "构建Docker镜像（首次运行需要几分钟）..."
    docker-compose build dashboard
    echo -e "${GREEN}✓ 镜像构建完成${NC}"
}

# 启动服务
start_services() {
    echo ""
    echo "启动服务..."
    docker-compose up -d dashboard
    echo -e "${GREEN}✓ 服务启动完成${NC}"
}

# 显示使用说明
show_usage() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║                    部署完成!                          ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${GREEN}访问地址:${NC}"
    echo "  可视化仪表板: http://localhost:8501"
    echo ""
    echo -e "${YELLOW}使用说明:${NC}"
    echo "  1. 将CIC-IDS-2017数据集放入 data/raw/ 目录"
    echo "  2. 在仪表板中点击'加载数据'按钮"
    echo ""
    echo -e "${YELLOW}其他命令:${NC}"
    echo "  启动仪表板:     docker-compose up -d dashboard"
    echo "  运行训练:       docker-compose run --rm train"
    echo "  数据预处理:     docker-compose run --rm preprocess"
    echo "  启动Jupyter:    docker-compose --profile dev up -d jupyter"
    echo "  启动TensorBoard: docker-compose --profile monitoring up -d tensorboard"
    echo "  停止所有服务:   docker-compose down"
    echo "  查看日志:       docker-compose logs -f dashboard"
    echo ""
}

# 主函数
main() {
    case "${1:-deploy}" in
        deploy)
            check_docker
            create_directories
            build_image
            start_services
            show_usage
            ;;
        stop)
            echo "停止所有服务..."
            docker-compose down
            echo -e "${GREEN}✓ 服务已停止${NC}"
            ;;
        restart)
            echo "重启服务..."
            docker-compose restart dashboard
            echo -e "${GREEN}✓ 服务已重启${NC}"
            ;;
        logs)
            docker-compose logs -f dashboard
            ;;
        train)
            echo "运行训练流程..."
            docker-compose run --rm train
            ;;
        help)
            echo "使用方法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  deploy   - 部署并启动服务 (默认)"
            echo "  stop     - 停止所有服务"
            echo "  restart  - 重启服务"
            echo "  logs     - 查看日志"
            echo "  train    - 运行训练"
            echo "  help     - 显示帮助"
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
