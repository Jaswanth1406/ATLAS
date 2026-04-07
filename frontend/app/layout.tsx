import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ATLAS — Road Segmentation",
  description:
    "Adaptive Thresholding with Language-Augmented Sensing. AI-powered road segmentation using deep learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}

/* ---- Inline Navbar ---- */
function Navbar() {
  return (
    <nav className="navbar">
      <a href="/" className="navbar-brand">
        <span className="logo-icon">🗺️</span>
        ATLAS
      </a>
      <div className="navbar-links">
        <a href="/">Home</a>
        <a href="/segment">Segment</a>
        <a href="/compare">Compare</a>
        <a href="/batch">Batch</a>
        <a href="/about">About</a>
        <a href="/segment" className="nav-cta">
          Try Now
        </a>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p className="footer-brand">🗺️ ATLAS — Mapping Roads with Intelligence</p>
      <p>
        Built by Divya R, Haripriya K &amp; Jaswanth Prasanna V &bull; IIT
        Madras &bull; 2026
      </p>
    </footer>
  );
}
