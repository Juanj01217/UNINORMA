"use client";

interface Props {
  models: string[];
  selected: string;
  onSelect: (model: string) => void;
  loading: boolean;
}

export default function ModelSelector({ models, selected, onSelect, loading }: Props) {
  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-gray-600 whitespace-nowrap">Modelo SLM:</label>
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        disabled={loading}
        className="text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
