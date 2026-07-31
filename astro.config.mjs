import { defineConfig } from 'astro/config';

// 中文为主：默认语言 zh（无前缀），英文在 /en/ 下
// base = 项目页子路径（因为 naphjohn.github.io 用户页已被 Gmeek 博客占用，Astro 放新仓库作项目页）
export default defineConfig({
  site: 'https://naphjohn.github.io',
  base: '/spec-decode-blog/',
  i18n: {
    defaultLocale: 'zh',
    locales: ['zh', 'en'],
    routing: {
      prefixDefaultLocale: false,
    },
  },
});
