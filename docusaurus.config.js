// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'AI Infrastructure', 
  tagline: 'Compute Optimization & Infrastructure Stack',
  favicon: 'img/favicon.ico',

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  future: {
    v4: true, 
  },

  // --- GITHUB PAGES CONFIGURATION ---
  // If your URL is username.github.io/ai-infra-docs/, baseUrl should be '/ai-infra-docs/'
  url: 'https://your-GitHub-username.github.io', 
  baseUrl: '/ai-infra-docs/', 
  organizationName: 'your-GitHub-username', 
  projectName: 'ai-infra-docs', 
  trailingSlash: false, // Recommended for GitHub Pages
  // ----------------------------------

  onBrokenLinks: 'throw',

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
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: '',
        logo: {
          alt: 'Site Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Infrastructure Guide',
          },
          {
            href: 'https://github.com/your-GitHub-username/ai-infra-docs',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Content',
            items: [
              {
                label: 'Overview',
                to: '/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI Infrastructure Project. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;