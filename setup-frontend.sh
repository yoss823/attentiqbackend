
#!/bin/bash

# ============================================================================
# SETUP FRONTEND - Attentiq
# Ce script crée la structure complète frontend + réorganise le backend
# ============================================================================

set -e

echo "🚀 Démarrage du setup..."

# ============================================================================
# 1. CRÉER LE DOSSIER BACKEND ET DÉPLACER LES FICHIERS
# ============================================================================

echo "📁 Création du dossier backend/"
mkdir -p backend

# Déplacer les fichiers backend
mv main.py backend/ 2>/dev/null || true
mv requirements.txt backend/ 2>/dev/null || true
mv Dockerfile backend/ 2>/dev/null || true
mv railway.json backend/ 2>/dev/null || true

echo "✅ Backend réorganisé"

# ============================================================================
# 2. CRÉER LA STRUCTURE FRONTEND
# ============================================================================

echo "📁 Création de la structure frontend/"

mkdir -p frontend/app/analyze
mkdir -p frontend/app/checkout/rapport-complet
mkdir -p frontend/app/checkout/5-rapports
mkdir -p frontend/app/checkout/illimite
mkdir -p frontend/app/merci
mkdir -p frontend/components
mkdir -p frontend/lib
mkdir -p frontend/public

echo "✅ Dossiers frontend créés"

# ============================================================================
# 3. CRÉER LES FICHIERS FRONTEND
# ============================================================================

echo "📝 Création des fichiers frontend..."

# frontend/package.json
cat > frontend/package.json << 'EOF'
{
  "name": "attentiq-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next": "^14.0.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@types/node": "^20.0.0",
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
EOF

# frontend/tsconfig.json
cat > frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": ".",
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
EOF

# frontend/next.config.js
cat > frontend/next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
};

module.exports = nextConfig;
EOF

# frontend/tailwind.config.ts
cat > frontend/tailwind.config.ts << 'EOF'
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
export default config
EOF

# frontend/postcss.config.js
cat > frontend/postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

# frontend/app/globals.css
cat > frontend/app/globals.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
EOF

# frontend/app/layout.tsx
cat > frontend/app/layout.tsx << 'EOF'
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Attentiq - Analyse de rétention TikTok',
  description: 'Analysez la rétention d\'audience de vos vidéos TikTok avec IA',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="fr">
      <body>
        {children}
      </body>
    </html>
  )
}
EOF

# frontend/app/page.tsx
cat > frontend/app/page.tsx << 'EOF'
'use client'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-5xl font-bold text-white mb-6">
          Attentiq
        </h1>
        <p className="text-xl text-slate-300 mb-8">
          Analysez la rétention d'audience de vos vidéos TikTok avec IA
        </p>
        <a
          href="/analyze"
          className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition"
        >
          Commencer l'analyse
        </a>
      </div>
    </main>
  )
}
EOF

# frontend/app/analyze/page.tsx
cat > frontend/app/analyze/page.tsx << 'EOF'
'use client'

import { useState } from 'react'

export default function AnalyzePage() {
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    // TODO: Appel API backend
    setLoading(false)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <div className="container mx-auto px-4 py-20">
        <h1 className="text-4xl font-bold text-white mb-8">
          Analyser une vidéo TikTok
        </h1>
        
        <form onSubmit={handleSubmit} className="max-w-2xl">
          <input
            type="url"
            placeholder="Collez l'URL de votre vidéo TikTok"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full px-4 py-3 rounded-lg mb-4 text-black"
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg transition disabled:opacity-50"
          >
            {loading ? 'Analyse en cours...' : 'Analyser'}
          </button>
        </form>
      </div>
    </main>
  )
}
EOF

# frontend/lib/api.ts
cat > frontend/lib/api.ts << 'EOF'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
})

export const analyzeVideo = async (url: string) => {
  const response = await api.post('/analyze', {
    url,
    platform: 'tiktok',
  })
  return response.data
}

export const getJobStatus = async (jobId: string) => {
  const response = await api.get(`/analyze/${jobId}`)
  return response.data
}
EOF

# frontend/lib/constants.ts
cat > frontend/lib/constants.ts << 'EOF'
export const PRICING_PLANS = [
  {
    id: 'rapport-complet',
    name: 'Rapport Complet',
    price: 9.99,
    features: ['1 analyse complète', 'Diagnostic détaillé'],
  },
  {
    id: '5-rapports',
    name: '5 Rapports',
    price: 39.99,
    features: ['5 analyses', 'Diagnostic détaillé', 'Support email'],
  },
  {
    id: 'illimite',
    name: 'Illimité',
    price: 99.99,
    features: ['Analyses illimitées', 'Support prioritaire', 'API access'],
  },
]
EOF

# frontend/.gitignore
cat > frontend/.gitignore << 'EOF'
# Dependencies
node_modules/
.pnp
.pnp.js

# Testing
coverage/

# Next.js
.next/
out/

# Production
build/

# Misc
.DS_Store
*.pem

# Debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Local env files
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
EOF

echo "✅ Fichiers frontend créés"

# ============================================================================
# 4. CRÉER LE .gitignore ROOT
# ============================================================================

cat > .gitignore << 'EOF'
# Backend
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Frontend
frontend/node_modules/
frontend/.next/
frontend/out/
frontend/.env.local
frontend/.env.development.local
frontend/.env.test.local
frontend/.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Misc
.env
.env.local
*.log
EOF

echo "✅ .gitignore créé"

# ============================================================================
# 5. ADAPTER railway.json POUR LE BACKEND
# ============================================================================

echo "🔧 Adaptation de railway.json..."

cat > backend/railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "numReplicas": 1,
    "startCommand": "python main.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30
  }
}
EOF

echo "✅ railway.json adapté"

# ============================================================================
# 6. RÉSUMÉ
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ SETUP TERMINÉ !"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Structure créée :"
echo "   ✅ backend/          (FastAPI existant)"
echo "   ✅ frontend/         (Next.js nouveau)"
echo "   ✅ .gitignore        (root)"
echo ""
echo "🚀 Prochaines étapes :"
echo "   1. git add ."
echo "   2. git commit -m 'feat: add frontend structure (Next.js + Tailwind)'"
echo "   3. git push origin main"
echo ""
echo "📋 Après le push :"
echo "   - Configure 2 services Railway :"
echo "     • Service 1 : Backend (rootDirectory: backend)"
echo "     • Service 2 : Frontend (rootDirectory: frontend)"
echo ""
echo "════════════════════════════════════════════════════════════════"
EOF
