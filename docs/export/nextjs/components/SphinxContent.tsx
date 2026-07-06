"use client";

import parse, {
  type DOMNode,
  type HTMLReactParserOptions,
  Element,
  domToReact,
} from "html-react-parser";
import Link from "next/link";

type SphinxContentProps = {
  html: string;
  className?: string;
};

/**
 * Renders Sphinx-exported HTML using your site's components and styles.
 *
 * Expects HTML already passed through `prepareSphinxBody()` so internal links
 * point at your Next.js routes and code blocks are flattened.
 */
export function SphinxContent({ html, className }: SphinxContentProps) {
  const options: HTMLReactParserOptions = {
    replace(domNode) {
      if (!(domNode instanceof Element)) {
        return undefined;
      }

      // Use Next.js Link for internal docs navigation
      if (domNode.name === "a") {
        const href = domNode.attribs.href ?? "";
        if (href.startsWith("/docs/")) {
          return (
            <Link href={href} className="text-brand underline-offset-2 hover:underline">
              {domToReact(domNode.children as DOMNode[], options)}
            </Link>
          );
        }
      }

      // Semantic code blocks after cleanup
      if (domNode.name === "pre") {
        return (
          <pre className="overflow-x-auto rounded-lg bg-muted p-4 text-sm">
            {domToReact(domNode.children as DOMNode[], options)}
          </pre>
        );
      }

      if (domNode.name === "code") {
        const parent = domNode.parent;
        const isBlock = parent instanceof Element && parent.name === "pre";
        if (isBlock) {
          return (
            <code className="font-mono text-sm">
              {domToReact(domNode.children as DOMNode[], options)}
            </code>
          );
        }
        return (
          <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm">
            {domToReact(domNode.children as DOMNode[], options)}
          </code>
        );
      }

      if (domNode.name === "table") {
        return (
          <div className="my-6 overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              {domToReact(domNode.children as DOMNode[], options)}
            </table>
          </div>
        );
      }

      return undefined;
    },
  };

  return (
    <article className={className ?? "prose prose-neutral max-w-none dark:prose-invert"}>
      {parse(html, options)}
    </article>
  );
}
