// 收集全部博客文章 frontmatter，按 (lang, series) 分组，
// 以 pubDate + slug 数字前缀排序，提供「上一篇 / 下一篇」查询。
// 供 src/layouts/BlogPost.astro 在每篇文章底部渲染阅读导航。
//
// 说明：刻意用 node:fs 直接读取磁盘上的 .md frontmatter，而不用
// import.meta.glob 导入页面模块 —— 否则会形成
//   布局 → 本模块 → .md 页面 → 布局
// 的循环依赖，导致打包时 frontmatter 尚未初始化而报错。

import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// 定位博客源目录：优先用 process.cwd()（Astro 在 dev/build 时均以项目根为 cwd），
// 回退到本文件所在目录（未打包的开发态）。Vite 打包后 import.meta.url 指向 dist，
// 不能再用它推导 src 路径，故以 cwd 为主。
const here = dirname(fileURLToPath(import.meta.url));
const root = process.cwd();
function resolveDir(...segs) {
  const cands = [join(root, ...segs), join(here, '..', ...segs)];
  for (const c of cands) {
    try {
      readdirSync(c);
      return c;
    } catch {
      /* try next */
    }
  }
  return cands[0];
}

const BLOG_ZH = resolveDir('src', 'pages', 'blog');
const BLOG_EN = resolveDir('src', 'pages', 'en', 'blog');

function parseFrontmatter(file) {
  const text = readFileSync(file, 'utf-8');
  const fm = {};
  const lines = text.split('\n');
  if (lines[0].trim() !== '---') return fm;
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === '---') break;
    const m = line.match(/^(title|pubDate|series|lang):\s*(.*)$/);
    if (m) fm[m[1]] = m[2].trim();
  }
  return fm;
}

function collect(dir, lang) {
  let entries = [];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  return entries
    .filter((f) => f.endsWith('.md'))
    .map((f) => {
      const fm = parseFrontmatter(join(dir, f));
      return {
        slug: f.replace(/\.md$/, ''),
        lang,
        title: fm.title || f.replace(/\.md$/, ''),
        pubDate: fm.pubDate || '',
        series: fm.series || '',
      };
    });
}

const ALL = [...collect(BLOG_ZH, 'zh'), ...collect(BLOG_EN, 'en')];

// 按 (lang, series) 分组并排序
const GROUPS = {};
for (const p of ALL) {
  const key = p.lang + '||' + p.series;
  (GROUPS[key] = GROUPS[key] || []).push(p);
}
for (const k in GROUPS) {
  GROUPS[k].sort((a, b) => {
    if (a.pubDate !== b.pubDate) return a.pubDate < b.pubDate ? -1 : 1;
    const na = leadingNum(a.slug);
    const nb = leadingNum(b.slug);
    if (na !== nb) return na - nb;
    return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
  });
}

// 提取 slug 中首个连续数字，用于同 pubDate 内的稳定排序（fw1<fw2<…<fw9、sys9<sys10<sys12）
function leadingNum(slug) {
  const m = slug.match(/\d+/);
  return m ? parseInt(m[0], 10) : -1;
}

// 系列阅读顺序（对齐首页板块排列）。
// 跨系列串联：系列末篇的「下一篇」接下一系列首篇，系列首篇的「上一篇」接上一系列末篇。
// 「每日AI热点」是时效新闻流（40+ 期、持续追加），刻意不参与跨系列串联，
// 只保留系列内按日期的 prev/next；未列出的系列同样不跨链。
const SERIES_ORDER = {
  zh: [
    '推测解码手记',
    '多模态解码手记',
    'vLLM 与 SGLang 框架解码手记',
    '推理系统基础设施手记',
    '智能体手记',
    'VLA 解码手记',
    '前沿架构解码手记',
    '算子讲解手记',
    '论文科普手记',
    '社区跟踪手记',
    '强化学习训练手记',
  ],
  en: [
    'Speculative Decoding Notes',
    'Multimodal Decoding Notes',
    'vLLM & SGLang Serving Notes',
    'Inference Systems Infrastructure Notes',
    'Agent Notes',
    'VLA Decoding Notes',
    'VLA Notes',
    'Frontier Architecture Decoding Notes',
    'Operator Notes',
    'Operator Explainers',
    'Paper Primer Notes',
    'Community Tracker Notes',
    'RL Training Notes',
  ],
};

// 给定当前文章的 (lang, series, slug)，返回相邻篇目。
// 系列内优先；系列首/末篇跨界接上一/下一系列（cross 标记供模板区分展示）。
export function getAdjacent(lang, series, slug) {
  const key = lang + '||' + (series || '');
  const group = GROUPS[key] || [];
  const idx = group.findIndex((p) => p.slug === slug);
  if (idx === -1) return { prev: null, next: null };
  let prev = idx > 0 ? { ...group[idx - 1], cross: false } : null;
  let next = idx < group.length - 1 ? { ...group[idx + 1], cross: false } : null;

  const order = SERIES_ORDER[lang] || [];
  const rank = order.indexOf(series || '');
  if (rank !== -1) {
    if (!next && rank < order.length - 1) {
      const ng = GROUPS[lang + '||' + order[rank + 1]];
      if (ng && ng.length) next = { ...ng[0], cross: true };
    }
    if (!prev && rank > 0) {
      const pg = GROUPS[lang + '||' + order[rank - 1]];
      if (pg && pg.length) prev = { ...pg[pg.length - 1], cross: true };
    }
  }
  return { prev, next };
}

// 与站内其他链接保持一致的 URL 写法（不含尾部斜杠，由 Astro 解析）
export function buildHref(base, lang, slug) {
  return lang === 'zh' ? `${base}blog/${slug}` : `${base}en/blog/${slug}`;
}
