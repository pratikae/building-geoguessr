# Beat Bob — Frontend

Next.js 16 game. See root `README.md` for full project docs.

## Run

```bash
npm install
npm run dev       # http://localhost:3000
```

Backend must be running at `localhost:8000` for live model predictions.
Falls back to mock data automatically if the backend is unreachable.

## Env

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000   # default
```
