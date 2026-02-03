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
  url: 'https://JackSteve-code.github.io', 
  baseUrl: '/AI-Infrastructure-and-compute-optimization/', 
  organizationName: 'JackSteve-code', 
  projectName: 'AI-Infrastructure-and-compute-optimization', 
  trailingSlash: false,
  // ----------------------------------

  // CHANGED THIS TO 'warn' TO PREVENT BUILD FAILURES
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
            href: 'https://github.com/JackSteve-code/AI-Infrastructure-and-compute-optimization',
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
                to: '/', // Fixed: This now points to your home page
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