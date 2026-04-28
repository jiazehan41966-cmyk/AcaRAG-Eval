import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AcaRAG Eval Dashboard",
  description: "Agentic RAG evaluation dashboard for academic papers"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main className="shell">
          <nav className="nav">
            <Link className="brand" href="/">
              AcaRAG Eval
            </Link>
            <div className="navlinks">
              <Link href="/">Dashboard</Link>
              <Link href="/evaluation-set">Evaluation Set</Link>
              <Link href="/documents">Documents</Link>
              <Link href="/chat">Chat</Link>
              <Link href="/evaluation">Eval Runs</Link>
              <Link href="/traces">Traces</Link>
            </div>
          </nav>
          {children}
        </main>
      </body>
    </html>
  );
}
