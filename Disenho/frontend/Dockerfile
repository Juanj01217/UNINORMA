# --- Etapa 1: Build ---
FROM node:20-alpine AS builder

WORKDIR /app

# NEXT_PUBLIC_API_URL=/api: el navegador siempre llama al proxy interno de Next.js
# El proxy reenvía al backend usando la variable BACKEND_URL (servidor → servidor)
ARG NEXT_PUBLIC_API_URL=/api
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- Etapa 2: Producción ---
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["npm", "start"]
