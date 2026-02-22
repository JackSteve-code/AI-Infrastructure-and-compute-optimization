// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'JACK.SYSTEMS', 
  tagline: 'Software and ML Engineer specializing in AI/ML infrastructure',
  favicon: 'img/favicon.ico',

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  url: 'https://JackSteve-code.github.io', 
  baseUrl: '/AI-Infrastructure-and-compute-optimization/', 
  organizationName: 'JackSteve-code', 
  projectName: 'AI-Infrastructure-and-compute-optimization', 
  trailingSlash: false,

  onBrokenLinks: 'warn', 
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/', 
        },
        blog: false, 
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      mermaid: {
        theme: {light: 'neutral', dark: 'forest'},
      },
      colorMode: {
        defaultMode: 'light',
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'JACK.SYSTEMS',
        logo: {
          alt: '',
          src: 'img/logo.svg',
          style: { display: 'none' }, // Hides the dinosaur logo
        },
        items: [
          {
            href: 'https://github.com/JackSteve-code/AI-Infrastructure-and-compute-optimization',
            label: 'GitHub Source',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'light',
        links: [],
        copyright: `© ${new Date().getFullYear()} Jack Steve | AI/ML Infrastructure Expert`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;