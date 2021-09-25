import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteExternalsPlugin } from 'vite-plugin-externals'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    viteExternalsPlugin({
      'onnxruntime-web': 'ort'
    })
  ],
  optimizeDeps: {
    exclude: [
      'onnxruntime-web'
    ]
  },
  base: '/mnist-demo/'
})
