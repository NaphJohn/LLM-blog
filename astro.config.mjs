import { defineConfig } from 'astro/config';

// 中文为主：默认语言 zh（无前缀），英文在 /en/ 下
export default defineConfig({
  site: 'https://naphjohn.github.io',
  i18n: {
    defaultLocale: 'zh',
    locales: ['zh', 'en'],
    routing: {
      prefixDefaultLocale: false,
    },
  },
});
