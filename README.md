# 大模型技术notes

基于 [Astro](https://astro.build) 的双语技术博客（**中文为主**，关键系列提供中英对照），部署于 GitHub Pages（`hangkaiwang.github.io`）。

## 技术栈
- Astro 5（静态站点，零运行时依赖）
- i18n：默认 `zh`（无前缀，`/`），英文在 `/en/`
- 部署：GitHub Actions 自动构建 → `gh-pages` artifact → Pages

## 本地预览
```bash
npm install
npm run dev      # http://localhost:4321
```

## 构建
```bash
npm run build    # 产物在 dist/
npm run preview  # 本地预览构建结果
```

## 目录结构
```
src/pages/index.astro          中文首页 / 文章列表
src/pages/en/index.astro       英文首页 / 文章列表
src/pages/blog/*.md            中文文章（推测解码手记）
src/pages/en/blog/*.md         英文文章
src/layouts/BlogPost.astro     文章布局
src/components/SiteHeader.astro 站点头（含中英切换 + 全局样式）
.github/workflows/deploy.yml   Pages 部署流水线
```

## 新增一篇（中英对照）
1. 在 `src/pages/blog/<slug>.md` 写中文，frontmatter 设 `altHref: /en/blog/<slug>`。
2. 在 `src/pages/en/blog/<slug>.md` 写英文，frontmatter 设 `altHref: /blog/<slug>`。
3. 在两语言首页的 `posts` 数组里各加一条。

## 推送到 GitHub Pages（需你在本机完成）
> 本仓库由 WorkBuddy 在本地生成并构建验证；由于运行环境没有 GitHub 凭证，推送需由你执行。

1. 在 GitHub **新建仓库 `hangkaiwang.github.io`**（用户页必须此名）。
2. 仓库 **Settings → Pages → Source 选 "GitHub Actions"**。
3. 在本目录执行：
   ```bash
   git remote add origin git@github.com:hangkaiwang/hangkaiwang.github.io.git
   git add .
   git commit -m "init: Astro bilingual blog (Ep1 + Ep5)"
   git branch -M main
   git push -u origin main
   ```
4. 等待 Actions 跑完（约 1–2 分钟），访问 https://hangkaiwang.github.io

> 若用 HTTPS 而非 SSH，把 remote 换成 `https://github.com/hangkaiwang/hangkaiwang.github.io.git`，推送时按提示登录 GitHub。
