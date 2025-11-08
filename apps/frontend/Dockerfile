# syntax=docker/dockerfile:1
FROM node:20-alpine AS builder
WORKDIR /app
COPY apps/frontend/package*.json ./
RUN npm install
COPY apps/frontend ./
RUN npm run build

FROM nginx:1.25-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
