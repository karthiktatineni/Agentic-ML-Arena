import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoML Arena - Autonomous AI Lab",
  description: "Autonomous Multi-Agent Machine Learning Cockpit and Stage Gate Protocol",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"
          rel="stylesheet"
        />
        <script src="https://cdn.tailwindcss.com"></script>
        <script
          id="tailwind-config"
          dangerouslySetInnerHTML={{
            __html: `
              tailwind.config = {
                darkMode: "class",
                theme: {
                  extend: {
                    colors: {
                      "outline": "#849495",
                      "on-error-container": "#ffdad6",
                      "on-background": "#e0e2eb",
                      "surface-bright": "#363940",
                      "inverse-on-surface": "#2d3037",
                      "on-tertiary": "#472a00",
                      "on-secondary-fixed-variant": "#5516be",
                      "tertiary": "#fff5ee",
                      "tertiary-container": "#ffd3a1",
                      "surface-variant": "#32353c",
                      "error-container": "#93000a",
                      "surface-container": "#1c2026",
                      "surface-tint": "#00dce6",
                      "on-tertiary-fixed-variant": "#653e00",
                      "surface-container-highest": "#32353c",
                      "on-primary-container": "#006a70",
                      "on-error": "#690005",
                      "on-primary-fixed": "#002022",
                      "on-surface-variant": "#b9cacb",
                      "secondary-container": "#571bc1",
                      "on-surface": "#e0e2eb",
                      "surface-container-lowest": "#0b0e14",
                      "surface-container-low": "#181c22",
                      "primary-fixed-dim": "#00dce6",
                      "on-secondary-fixed": "#23005c",
                      "on-secondary-container": "#c4abff",
                      "on-primary-fixed-variant": "#004f53",
                      "background": "#10131a",
                      "outline-variant": "#3a494b",
                      "inverse-primary": "#00696f",
                      "primary-container": "#00f2fe",
                      "on-secondary": "#3c0091",
                      "secondary-fixed-dim": "#d0bcff",
                      "on-tertiary-fixed": "#2a1700",
                      "on-primary": "#00373a",
                      "primary": "#e0fdff",
                      "surface-container-high": "#272a31",
                      "tertiary-fixed": "#ffddb8",
                      "error": "#ffb4ab",
                      "secondary": "#d0bcff",
                      "primary-fixed": "#6ff6ff",
                      "on-tertiary-container": "#875400",
                      "surface-dim": "#10131a",
                      "secondary-fixed": "#e9ddff",
                      "surface": "#10131a",
                      "inverse-surface": "#e0e2eb",
                      "tertiary-fixed-dim": "#ffb95f"
                    },
                    borderRadius: {
                      DEFAULT: "0.125rem",
                      lg: "0.25rem",
                      xl: "0.5rem",
                      full: "0.75rem"
                    },
                    spacing: {
                      "space-xl": "2rem",
                      "gutter-desktop": "1.5rem",
                      "gutter": "1rem",
                      "space-md": "0.75rem",
                      "space-lg": "1.25rem",
                      "space-sm": "0.5rem",
                      "space-xs": "0.25rem",
                      "margin-desktop": "2rem",
                      "margin": "1rem"
                    },
                    fontFamily: {
                      "body-sm": ["Outfit", "sans-serif"],
                      "body-md": ["Outfit", "sans-serif"],
                      "body-lg": ["Outfit", "sans-serif"],
                      "headline-md": ["Outfit", "sans-serif"],
                      "headline-lg": ["Outfit", "sans-serif"],
                      "headline-xl": ["Outfit", "sans-serif"],
                      "label-sm": ["JetBrains Mono", "monospace"],
                      "label-md": ["JetBrains Mono", "monospace"],
                      "label-lg": ["JetBrains Mono", "monospace"]
                    }
                  }
                }
              };
            `,
          }}
        />
      </head>
      <body className="bg-surface font-body-md text-body-md text-on-surface min-h-screen selection:bg-primary-container selection:text-on-primary-container">
        {children}
      </body>
    </html>
  );
}
