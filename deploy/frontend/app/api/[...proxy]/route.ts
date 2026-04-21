import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

// 5 minutos — el backend tiene su propio timeout de 90s en Ollama, por lo que
// el proxy solo deberia disparar si el backend tarda mas de lo esperado.
const PROXY_TIMEOUT_MS = 300_000;

async function proxyRequest(req: NextRequest) {
  const path = req.nextUrl.pathname.replace(/^\/api/, "");
  const url = `${BACKEND_URL}${path}${req.nextUrl.search}`;

  const headers: Record<string, string> = { "Content-Type": "application/json" };

  let body: string | undefined;
  if (req.method !== "GET" && req.method !== "HEAD") {
    try {
      body = await req.text();
    } catch {
      return NextResponse.json(
        { detail: "No se pudo leer el cuerpo de la solicitud." },
        { status: 400 }
      );
    }
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROXY_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: req.method,
      headers,
      body,
      signal: controller.signal,
    });

    const data = await res.text();
    return new NextResponse(data, {
      status: res.status,
      headers: {
        "Content-Type": res.headers.get("Content-Type") || "application/json",
      },
    });
  } catch (e: unknown) {
    const isAbort =
      e instanceof Error &&
      (e.name === "AbortError" || e.message.includes("aborted"));
    const message = isAbort
      ? "La consulta tardó demasiado. Prueba con un modelo más rápido o una pregunta más corta."
      : e instanceof Error
        ? e.message
        : "Backend no disponible";
    const status = isAbort ? 504 : 502;
    return NextResponse.json({ detail: message }, { status });
  } finally {
    clearTimeout(timer);
  }
}

export async function GET(req: NextRequest) {
  return proxyRequest(req);
}

export async function POST(req: NextRequest) {
  return proxyRequest(req);
}
