import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Spectral Image Segmentation Using Eigenvectors',
  description: 'Interactive Linear Algebra project: image pixels as a graph, Laplacian eigenvectors, and spectral clustering. Newsprint edition.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
