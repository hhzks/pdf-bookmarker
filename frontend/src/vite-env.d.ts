/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Unset in local dev, where vite.config.ts proxies /api to the backend.
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
