import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownRendererProps {
  children: string;
}

/**
 * MarkdownRenderer component renders markdown content with GitHub-flavored markdown (GFM) support.
 * Uses react-markdown with remark-gfm plugin for extended markdown features like tables, strikethrough, and task lists.
 * Applies Tailwind prose styling for consistent typography.
 */
const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ children }) => {
  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {children}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
