import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: './', // 👈 這行最重要，確保部署後圖片與程式路徑不會出錯
})