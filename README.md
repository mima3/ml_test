
## VSCodeを使う場合

**前提：**
Dev Containers 拡張機能を入れる。

**コンテナを開く方法**
Shift+cmd+p or 表示→コマンドパレット
Dev Containers: Rebuild and Reopen in Container

or

Dev Containers: Reopen in Container


## 使わないケース

```
# ビルド
docker compose -f docker-compose-dev.yml build

# 起動
docker compose -f docker-compose-dev.yml up -d

# 中に入る
docker compose -f docker-compose-dev.yml exec ml-cpu bash
```
