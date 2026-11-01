import type { RemedyCard as RemedyCardData } from '../services/chatApi'

type Props = {
  remedy: RemedyCardData
  selected: boolean
  disabled: boolean
  onSelect: () => void
}

export default function RemedyCard({ remedy, selected, disabled, onSelect }: Props) {
  return (
    <article className="mt-3 rounded-sm border border-[#dce0d5] bg-white/80 p-4 text-left">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="font-medium text-[#31543c]">{remedy.name}</h3>
        <span className="rounded-sm bg-[#edf0e8] px-2 py-1 text-[11px] text-[#53665a]">
          {remedy.kind === 'pdf' ? 'From uploaded PDFs' : 'AI-generated'}
        </span>
      </div>
      <p className="mt-2 text-sm leading-5 text-[#405244]">{remedy.description}</p>
      {remedy.ingredients && <p className="mt-2 text-xs leading-5 text-[#53665a]"><strong>Ingredients:</strong> {remedy.ingredients}</p>}
      {remedy.preparation && <p className="mt-1 text-xs leading-5 text-[#53665a]"><strong>Preparation:</strong> {remedy.preparation}</p>}
      {remedy.kind === 'ai_generated' && (
        <div className="mt-3 space-y-1 border-t border-[#e6e9e1] pt-3 text-xs leading-5 text-[#53665a]">
          {remedy.traditional_or_supportive_basis && <p><strong>Traditional/supportive basis:</strong> {remedy.traditional_or_supportive_basis}</p>}
          {remedy.evidence && <p><strong>Evidence:</strong> {remedy.evidence}</p>}
          {remedy.possible_benefits && <p><strong>Possible benefits:</strong> {remedy.possible_benefits}</p>}
          {remedy.side_effects && <p><strong>Side effects:</strong> {remedy.side_effects}</p>}
          {remedy.who_should_avoid && <p><strong>Who should avoid:</strong> {remedy.who_should_avoid}</p>}
          {remedy.precautions_interactions && <p><strong>Precautions/interactions:</strong> {remedy.precautions_interactions}</p>}
          {remedy.when_to_seek_care && <p><strong>When to seek care:</strong> {remedy.when_to_seek_care}</p>}
        </div>
      )}
      {remedy.sources.length > 0 && (
        <p className="mt-2 text-xs text-[#718071]">
          {remedy.sources.map((source) => `${source.source}, page ${source.page}`).join('; ')}
        </p>
      )}
      <button
        type="button"
        disabled={disabled}
        onClick={onSelect}
        className="mt-3 rounded-sm border border-[#31543c] px-3 py-2 text-xs font-medium text-[#31543c] hover:bg-[#edf0e8] disabled:cursor-wait disabled:opacity-60"
      >
        {selected ? 'View This Remedy Again' : 'View This Remedy'}
      </button>
    </article>
  )
}
