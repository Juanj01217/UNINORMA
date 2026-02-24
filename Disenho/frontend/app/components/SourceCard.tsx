import { Source } from "../lib/api";

interface Props {
  sources: Source[];
}

function cleanFilename(raw: string): string {
  try {
    return decodeURIComponent(raw.split("/").pop() || raw).replace(".pdf", "");
  } catch {
    return raw;
  }
}

export default function SourceCard({ sources }: Props) {
  if (!sources.length) return null;

  return (
    <div className="mt-3 pt-3 border-t border-gray-200">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
        Fuentes consultadas
      </p>
      <div className="flex flex-wrap gap-2">
        {sources.map((s, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 text-xs"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            {cleanFilename(s.title || s.source)}
            {s.page && <span className="text-blue-400">· pág. {s.page}</span>}
          </span>
        ))}
      </div>
    </div>
  );
}
