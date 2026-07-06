import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SphinxContent } from "@/components/SphinxContent";
import {
  DOCS_BASE_PATH,
  docsPathFromSlug,
  fetchManifest,
  fetchSphinxPage,
} from "@/lib/sphinx-docs";
import Link from "next/link";

type PageProps = {
  params: Promise<{ slug?: string[] }>;
};

function slugFromParams(slug?: string[]): string {
  if (!slug || slug.length === 0) {
    return "index";
  }
  return slug.join("/");
}

export async function generateStaticParams() {
  const manifest = await fetchManifest();
  return manifest.slugs.map((slug) => ({
    slug: slug === "index" ? [] : slug.split("/"),
  }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const pageSlug = slugFromParams(slug);
  const page = await fetchSphinxPage(pageSlug);
  if (!page) {
    return { title: "Not found" };
  }

  return {
    title: `${page.title} | SDK Docs`,
    description: `BioLM Python SDK documentation: ${page.title}`,
    alternates: {
      canonical: docsPathFromSlug(pageSlug),
    },
  };
}

export default async function SdkDocsPage({ params }: PageProps) {
  const { slug } = await params;
  const pageSlug = slugFromParams(slug);
  const [page, manifest] = await Promise.all([
    fetchSphinxPage(pageSlug),
    fetchManifest(),
  ]);

  if (!page) {
    notFound();
  }

  return (
    <div className="mx-auto flex max-w-6xl gap-10 px-6 py-10">
      <aside className="hidden w-56 shrink-0 lg:block">
        <nav className="sticky top-10 space-y-6 text-sm">
          {manifest.nav.map((section) => (
            <div key={section.caption ?? section.items[0]?.slug}>
              {section.caption ? (
                <p className="mb-2 font-semibold text-muted-foreground">
                  {section.caption}
                </p>
              ) : null}
              <ul className="space-y-1">
                {section.items.map((item) => (
                  <li key={item.slug}>
                    <Link
                      href={docsPathFromSlug(item.slug)}
                      className={
                        item.slug === pageSlug
                          ? "font-medium text-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      }
                    >
                      {item.title}
                    </Link>
                    {item.children?.length ? (
                      <ul className="ml-3 mt-1 space-y-1 border-l pl-3">
                        {item.children.map((child) => (
                          <li key={child.slug}>
                            <Link
                              href={docsPathFromSlug(child.slug)}
                              className={
                                child.slug === pageSlug
                                  ? "font-medium text-foreground"
                                  : "text-muted-foreground hover:text-foreground"
                              }
                            >
                              {child.title}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
      </aside>

      <main className="min-w-0 flex-1">
        <SphinxContent html={page.body} />

        <footer className="mt-12 flex justify-between border-t pt-6 text-sm">
          {page.prev ? (
            <Link
              href={docsPathFromSlug(page.prev.slug)}
              className="text-muted-foreground hover:text-foreground"
            >
              ← {page.prev.title}
            </Link>
          ) : (
            <span />
          )}
          {page.next ? (
            <Link
              href={docsPathFromSlug(page.next.slug)}
              className="text-muted-foreground hover:text-foreground"
            >
              {page.next.title} →
            </Link>
          ) : null}
        </footer>
      </main>
    </div>
  );
}

export const dynamic = "force-static";
