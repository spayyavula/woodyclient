import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // Prevent Vite from treating $ as a special character in string literals
    __DOLLAR_SIGN__: '"$"',
  },
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: ['lucide-react'],
  },
  build: {
    target: 'esnext',
    // Ensure proper handling of special characters during build
    rollupOptions: {
      output: {
        // Prevent mangling of dollar signs in output
        sanitizeFileName: (name) => name.replace(/[\$]/g, '_DOLLAR_'),
      },
    },
  }
});