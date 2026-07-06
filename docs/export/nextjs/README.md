# Next.js integration for Sphinx JSON docs

Reference code for rendering Sphinx ``-b json`` output in the main BioLM Next.js site.

## Build and publish (py-biolm)

CI on the ``production`` branch runs ``make docs-publish`` and deploys to GitHub Pages.

Local build:

```bash
pip install -e .
make docs-publish
```

Output: ``docs/_build/publish/`` containing only:

- ``manifest.json`` — navigation tree, titles, prev/next links, slug index
- ``**/*.fjson`` — one page per doc; main field is ``body`` (pre-rendered HTML)

## Published URLs (GitHub Pages)

gh-pages hosts **JSON only** at the site root, e.g. ``https://<org>.github.io/<repo>/``:

| Resource | URL |
|----------|-----|
| Manifest | ``https://<org>.github.io/<repo>/manifest.json`` |
| Home page | ``https://<org>.github.io/<repo>/index.fjson`` |
| Example | ``https://<org>.github.io/<repo>/getting-started/quickstart.fjson`` |

## Next.js configuration

Fetch at SSR/build time from gh-pages (do not copy files into ``public/``):

```
SDK_DOCS_ORIGIN=https://<org>.github.io/<repo>
SDK_DOCS_ASSET_BASE=
```

Fetch URLs:

- ``${SDK_DOCS_ORIGIN}/manifest.json``
- ``${SDK_DOCS_ORIGIN}/{slug}.fjson``

User-facing routes on the main site: ``/docs/sdk/[...slug]``

## Install in Next.js

```bash
npm install html-react-parser
```

## Reference files

| File | Purpose |
|------|---------|
| ``lib/sphinx-docs.ts`` | Fetch pages, rewrite internal links, clean code blocks |
| ``components/SphinxContent.tsx`` | Render Sphinx HTML with your own styles |
| ``app/docs/sdk/[...slug]/page.tsx`` | Example App Router page (SSR) |

## Rendering notes

- Ignore Sphinx CSS/theme — render ``body`` with your own components/styles
- Strip Pygments ``<span>`` noise from code blocks; use plain ``<pre><code>`` text
- Rewrite internal links from Sphinx paths (``.html`` suffixes, ``../``) to site routes (e.g. ``/docs/sdk/getting-started/concepts``)
- Use ``manifest.json`` for sidebar nav and ``generateStaticParams``
- Use each page's ``title`` for ``generateMetadata`` (SEO/agents)

## Styling

Style semantic elements (``h1``, ``p``, ``pre``, ``table``) with your design system or Tailwind Typography (``@tailwindcss/typography``).
