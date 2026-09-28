# Multi-stage: build the static bundle with Node, serve it with nginx --
# the shipped image has no Node runtime in it at all.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Vite env vars are baked in at build time, not read at container start --
# set this to wherever the API is actually reachable from the browser
# (docker-compose.yml sets it to http://localhost:8000 for local use).
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
