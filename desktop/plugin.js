/**
 * qdrant-index — desktop statusbar pill (runtime-loaded ESM, NO JSX).
 *
 * Loaded by the runtime pipeline (specifier rewrite -> SDK/react shim blobs ->
 * blob import -> register). Plain ESM js with jsx()/jsxs() calls — exactly what
 * the loader expects. Only allowed bare imports: @hermes/plugin-sdk and react*.
 *
 * Shows the focused project's Qdrant index state in the status bar (right),
 * VSCode-Git style (database icon + name + state marker, color = health,
 * details on hover):
 *   connected    -> db icon + "Index" (green)
 *   stale        -> db icon + "Index <N>~" (amber, N = changed + new)
 *   not indexed  -> db icon + "Index" (dimmed) — project has no collection yet
 *   disabled     -> db icon + "Index" (dimmed) — automatic indexing off
 *   offline      -> db icon + "Index" (dimmed) — backend unreachable
 *   no cwd       -> renders nothing
 * File count / collection / last indexed live in the popover.
 *
 * Session-relative: the pill tracks the focused project's cwd. When the
 * focused project changes it fires an incremental /refresh so the index
 * follows the open session (an unindexed project shows "not indexed", never
 * the previous project's green state).
 *
 * Click opens a details popover (collection, root, counts, last indexed) with a
 * "Reindex now" button (POST /reindex) and a ⚙️ Settings view with the
 * "Enable automatic indexing" master switch (PUT /config { enabled }) plus the
 * Qdrant server + embedding endpoint editor (GET/PUT /config, GET /config/test).
 * Polls every 30s via useQuery. REST errors (backend down) degrade to the
 * dimmed state — never crash.
 */

import {
  atom,
  Checkbox,
  cn,
  Codicon,
  host,
  Popover,
  PopoverContent,
  PopoverTrigger,
  relativeTime,
  STATUSBAR_AREAS,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import { useEffect, useRef, useState } from 'react'

// Single-instance interaction state (module-level: register() runs once).
const $open = atom(false)

// One labeled input row for the settings form.
function Field({ label, hint, ...props }) {
  return jsxs(
    'label',
    {
      className: 'flex flex-col gap-1',
      children: [
        jsx('span', { className: 'text-[0.6875rem] text-muted-foreground', children: label }),
        jsx('input', {
          className:
            'w-full rounded border border-(--ui-stroke-secondary) bg-(--ui-surface, transparent) px-2 py-1 text-[0.75rem] text-foreground outline-none focus:border-primary',
          ...props
        }),
        hint ? jsx('span', { className: 'text-[0.625rem] text-(--ui-text-quaternary, #9ca3af)', children: hint }) : null
      ]
    }
  )
}

export default {
  id: 'qdrant-index',
  name: 'Qdrant Index',
  description: 'Semantic index of the focused project — live file count + staleness, and server/embedding settings.',
  defaultEnabled: false,
  register(ctx) {
    const setOpen = (v) => $open.set(typeof v === 'boolean' ? v : !v)

    function QdrantPill() {
      const cwd = useValue(host.state.cwd)
      const focusedStoredId = useValue(host.state.focusedStoredSessionId)
      const open = useValue($open)
      const [view, setView] = useState('status') // 'status' | 'settings'
      const [form, setForm] = useState(null)
      const [busy, setBusy] = useState(false)
      const [enBusy, setEnBusy] = useState(false) // Enable-checkbox busy (separate from Save/Test)
      const [testResult, setTestResult] = useState(null)
      // Local flag set the instant the user clicks Reindex/Index, cleared once
      // the backend reports the op finished. Keeps the button disabled+labelled
      // from the click without depending on isFetching (which flickers on every
      // poll).
      const [manualBusy, setManualBusy] = useState(false)

      // ---- Session-switch race guard -----------------------------------
      // The focused stored id flips the instant the user clicks another
      // session, but host.state.cwd only settles once that session's resume
      // lands — in the gap it still points at the PREVIOUS session's project.
      // While switching we freeze the pill (no status query, no /refresh) so
      // we never read or index the wrong project. We leave the window when the
      // new cwd lands (cwd change) or after a grace timeout (switch between
      // two sessions of the SAME project, where cwd never changes).
      const [transitioning, setTransitioning] = useState(false)
      const switchTimerRef = useRef(null)
      const prevFocusedIdRef = useRef(focusedStoredId)
      useEffect(() => {
        if (prevFocusedIdRef.current === focusedStoredId) return
        prevFocusedIdRef.current = focusedStoredId
        setTransitioning(true)
        if (switchTimerRef.current) clearTimeout(switchTimerRef.current)
        switchTimerRef.current = setTimeout(() => setTransitioning(false), 800)
        return () => {
          if (switchTimerRef.current) clearTimeout(switchTimerRef.current)
        }
      }, [focusedStoredId])
      // The new workspace landed → settle immediately (no need to wait the
      // grace window).
      useEffect(() => {
        if (!transitioning) return
        setTransitioning(false)
        if (switchTimerRef.current) {
          clearTimeout(switchTimerRef.current)
          switchTimerRef.current = null
        }
      }, [cwd]) // eslint-disable-line react-hooks/exhaustive-deps

      const { data, refetch } = useQuery({
        queryKey: ['qdrant', 'status', cwd],
        queryFn: () => ctx.rest('/status?root=' + encodeURIComponent(cwd)),
        enabled: !!cwd && !transitioning,
        refetchInterval: 30000,
        // The shared app QueryClient defaults staleTime to 60_000. React Query
        // v5's shouldFetchOptionally gates on isStale, so with the 60s global
        // default, a session switch or transition-end (enabled false→true)
        // does NOT trigger an immediate refetch when data is <60s old — the
        // pill shows stale/previous-project state until the next 30s tick.
        // Override to 0 so the query is always stale and refetches immediately
        // on mount, session switch, and transition end.
        staleTime: 0,
        // The interval callback gates on focusManager.isFocused() by default,
        // so the 30s poll pauses entirely when the window loses focus. If a
        // reindex completes while the user is in another app, the pill stays
        // stale until they refocus. Keep polling in the background so the pill
        // reflects the latest index state regardless of window focus.
        refetchIntervalInBackground: true
      })

      const { data: cfg, refetch: refetchCfg } = useQuery({
        queryKey: ['qdrant', 'config'],
        queryFn: () => ctx.rest('/config'),
        enabled: view === 'settings'
      })

      // Seed the form once config arrives (and don't clobber while typing).
      useEffect(() => {
        if (cfg && !form) {
          setForm({
            enabled: cfg.enabled !== false,
            host: cfg.qdrant.host,
            port: String(cfg.qdrant.port),
            base_url: cfg.embedding.base_url,
            model: cfg.embedding.model,
            api_key: '', // blank = keep current (GET redacts the real key)
            vector_dim: String(cfg.embedding.vector_dim)
          })
        }
      }, [cfg]) // eslint-disable-line react-hooks/exhaustive-deps

      // When the focused project (session cwd) changes, bring that project's
      // index up to date automatically — it may be stale (edited in another
      // session) or not indexed yet. Incremental (no force): only changed/new
      // files are re-embedded, so switching back and forth is cheap.
      const prevCwdRef = useRef(null)
      useEffect(() => {
        const prev = prevCwdRef.current
        if (!cwd) {
          prevCwdRef.current = null
          return
        }
        // Mid-switch: the cwd may still be the previous project's. Don't
        // refresh (or register it as "handled") until the transition settles.
        if (transitioning) {
          return
        }
        if (prev === cwd) return
        prevCwdRef.current = cwd
        // Master switch off: the user opted out of automatic indexing —
        // show the (possibly stale) state, but don't index on our own.
        if (data && data.enabled === false) return
        ctx.rest('/refresh?root=' + encodeURIComponent(cwd), { method: 'POST' })
          .then(() => {
            // Poll a few times so the pill reflects the refreshed state
            // without waiting for the 30s interval.
            let tries = 0
            const timer = setInterval(() => {
              tries += 1
              refetch().then(() => {
                if (tries >= 10) clearInterval(timer)
              })
            }, 3000)
          })
          .catch(() => { /* refresh is best-effort */ })
      }, [cwd, transitioning]) // eslint-disable-line react-hooks/exhaustive-deps

      // Home/detached: the backend refuses to index the home directory
      // (sessions in the Home bucket fall back to ~/ as cwd). Render
      // nothing — there is no project to show.
      if (!cwd) {
        return null
      }
      if (data && data.indexable === false) {
        return null
      }

      // Keep the response object while the focused workspace is settled.
      // `data && !transitioning` returns the boolean `true` when data exists,
      // which makes live.indexed/live.file_count undefined in the renderer.
      const live = !transitioning ? data : null
      const stale = !!live && live.stale
      const indexed = !!live && live.indexed
      const disabled = !!live && live.enabled === false
      // Collection was deleted out-of-band: local cache still claims indexed,
      // but the live check downgraded it. Amber = attention + rebuildable
      // (distinct from dimmed = never indexed).
      const deleted = !!live && !live.indexed && (live.collection_state === 'missing' || live.collection_state === 'empty')
      const n = live ? (live.changed || 0) + (live.new || 0) : 0
      const count = live ? live.file_count : null

      const pillLabel = 'Index'

      // Live pipeline progress: the backend writes files_done/files_total into
      // the running op after each file is checkpointed; /status (polled every
      // 3s) carries them here. Declared BEFORE `tip` — const is in the TDZ
      // until this line, so any earlier read throws ReferenceError.
      const opProg = live && live.last_op && live.last_op.status === 'running' && live.last_op.files_total
        ? `${live.last_op.files_done || 0}/${live.last_op.files_total}`
        : null

      const tip = transitioning
        ? 'Resolving project…'
        : !live
          ? 'Qdrant index not available for this project'
          : disabled
            ? 'Disabled — automatic indexing off. Click to enable or index manually.'
            : !indexed
              ? live.collection_state === 'missing' || live.collection_state === 'empty'
                ? `Collection '${live.collection}' was deleted from Qdrant. Click to rebuild it.`
                : 'Not indexed. Click to index this project.'
              : stale
                ? `Stale — ${n} file(s) changed/new. Click for details.`
                : `Connected — ${count} file(s) indexed. Click for details.`
      if (opProg) tip = `Indexing… ${opProg} files done`

      const lastIndexed = live && live.last_indexed ? relativeTime(new Date(live.last_indexed).getTime()) : null
      // Busy = a background index op is actually running (reindexing/last_op),
      // OR the user just clicked and we're waiting for the first post-POST
      // refetch to confirm. NOT isFetching: that flips true/false on every
      // poll and made the footer buttons flicker. These sources are stable.
      const reindexBusy =
        manualBusy ||
        !!(live && (live.reindexing || (live.last_op && live.last_op.status === 'running')))

      const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))

      const save = () => {
        setBusy(true)
        const body = {
          qdrant: { host: form.host.trim(), port: parseInt(form.port, 10) },
          embedding: {
            base_url: form.base_url.trim(),
            model: form.model.trim(),
            vector_dim: parseInt(form.vector_dim, 10)
          }
        }
        if (form.api_key.trim()) body.embedding.api_key = form.api_key.trim()
        ctx.rest('/config', { method: 'PUT', body })
          .then(() => {
            refetchCfg()
            setTestResult(null)
            host.notify({ kind: 'success', title: 'Qdrant settings saved', message: 'Server & embedding endpoint updated (applies to the next operation).' })
          })
          .catch((e) => host.notifyError(e, 'Qdrant: save failed'))
          .finally(() => setBusy(false))
      }

      const test = () => {
        // Save first so the probe uses the values on screen, then probe.
        setBusy(true)
        const body = {
          qdrant: { host: form.host.trim(), port: parseInt(form.port, 10) },
          embedding: {
            base_url: form.base_url.trim(),
            model: form.model.trim(),
            vector_dim: parseInt(form.vector_dim, 10)
          }
        }
        if (form.api_key.trim()) body.embedding.api_key = form.api_key.trim()
        ctx.rest('/config', { method: 'PUT', body })
          .then(() => ctx.rest('/config/test'))
          .then((res) => setTestResult(res))
          .catch((e) => host.notifyError(e, 'Qdrant: test failed'))
          .finally(() => setBusy(false))
      }

      // Master switch — PUT /config { enabled } works standalone (no other
      // keys), so the checkbox applies immediately without a Save click.
      // The checkbox is controlled by form.enabled, and the config seed
      // effect only runs while form === null — so a plain refetchCfg()
      // never pushes the new value into the checkbox: it visually stayed on
      // the OLD state until the pill remounted (reopening the popup in a
      // fresh mount). Flip the form state optimistically on click and roll
      // it back if the PUT fails.
      const toggleEnabled = (next) => {
        setEnBusy(true)
        setForm((f) => (f ? { ...f, enabled: next } : f))
        ctx.rest('/config', { method: 'PUT', body: { enabled: next } })
          .then(() => {
            refetchCfg()
            refetch()
            host.notify({
              kind: next ? 'success' : 'info',
              title: next ? 'Qdrant indexing enabled' : 'Qdrant indexing disabled',
              message: next
                ? 'Automatic index refresh is back on.'
                : 'No automatic indexing — use "Index now" to refresh manually.'
            })
          })
          .catch((e) => {
            setForm((f) => (f ? { ...f, enabled: !next } : f))
            host.notifyError(e, 'Qdrant: enable toggle failed')
          })
          .finally(() => setEnBusy(false))
      }

      const footerBtn = (label, kind, onClick, disabled) =>
        jsx(
          'button',
          {
            type: 'button',
            disabled,
            onClick,
            className: cn(
              'rounded px-2 py-1 text-[0.75rem]',
              kind === 'primary'
                ? 'font-medium bg-primary text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40'
                : 'text-muted-foreground hover:bg-(--chrome-action-hover) disabled:opacity-40'
            ),
            children: label
          }
        )

      return jsx(Popover, {
        open,
        onOpenChange: setOpen,
        children: [
          jsx(
            PopoverTrigger,
            {
              asChild: true,
              children: jsx(
                'button',
                {
                  type: 'button',
                  title: tip,
                  className: cn(
                    'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] tabular-nums transition-colors',
                    indexed && !stale && !disabled
                      ? 'text-emerald-400 hover:bg-(--chrome-action-hover)'
                      : (stale || deleted) && !disabled
                        ? 'text-amber-500 hover:bg-(--chrome-action-hover)'
                        : 'text-(--ui-text-quaternary, #6b7280) hover:bg-(--chrome-action-hover)'
                  ),
                  children: [
                    // Database glyph — inherits the pill's state color (the
                    // codicon is currentColor). Spins while an index op runs.
                    jsx(Codicon, { name: 'database', size: '0.8125rem', spinning: reindexBusy }),
                    jsx('span', { children: pillLabel }),
                    opProg ? jsx('span', { className: 'tabular-nums text-amber-500', children: ` ${opProg}` }) : null,
                    stale && !disabled ? jsx('span', { className: 'text-amber-500', children: ` ${n}~` }) : null
                  ]
                }
              )
            }
          ),
          jsx(
            PopoverContent,
            {
              align: 'end',
              side: 'top',
              sideOffset: 8,
              collisionPadding: 8,
              className: view === 'settings' ? 'w-80' : 'w-72',
              children: [
                view === 'settings'
                  ? jsxs('div', {
                      className: 'flex flex-col gap-3 text-[0.8125rem]',
                      children: [
                        jsxs('div', {
                          className: 'flex items-center justify-between',
                          children: [
                            jsx('h3', { className: 'text-[0.8125rem] font-semibold', children: 'Server & embedding' }),
                            footerBtn('✕ close', 'ghost', () => { setView('status'); setTestResult(null) })
                          ]
                        }),
                        !form
                          ? jsx('div', { className: 'text-[0.75rem] text-muted-foreground', children: 'Loading…' })
                          : jsxs('div', {
                              className: 'flex flex-col gap-3',
                              children: [
                                // Master switch — applies on click (PUT /config
                                // { enabled }), no Save needed.
                                jsxs(
                                  'label',
                                  {
                                    className: 'flex cursor-pointer items-center gap-2 rounded border border-(--ui-stroke-secondary) px-2 py-1.5',
                                    children: [
                                      jsx(Checkbox, {
                                        checked: form.enabled,
                                        disabled: enBusy,
                                        onCheckedChange: (v) => toggleEnabled(v === true)
                                      }),
                                      jsxs('div', {
                                        className: 'flex flex-col',
                                        children: [
                                          jsx('span', { className: 'text-[0.75rem] font-medium', children: 'Enable automatic indexing' }),
                                          jsx('span', { className: 'text-[0.625rem] text-muted-foreground', children: 'Refresh the index after edits; manual "Index now" always works' })
                                        ]
                                      })
                                    ]
                                  }
                                ),
                                jsxs('div', {
                                  className: 'flex flex-col gap-2',
                                  children: [
                                    jsx('div', { className: 'text-[0.6875rem] font-medium uppercase tracking-wide text-muted-foreground', children: 'Qdrant server' }),
                                    Field({ label: 'Host', value: form.host, onChange: set('host'), placeholder: 'localhost' }),
                                    Field({ label: 'Port', type: 'number', value: form.port, onChange: set('port'), placeholder: '6333' })
                                  ]
                                }),
                                jsxs('div', {
                                  className: 'flex flex-col gap-2',
                                  children: [
                                    jsx('div', { className: 'text-[0.6875rem] font-medium uppercase tracking-wide text-muted-foreground', children: 'Embedding endpoint' }),
                                    Field({ label: 'Base URL', value: form.base_url, onChange: set('base_url'), placeholder: 'http://host:port/v1' }),
                                    jsxs('div', {
                                      className: 'grid grid-cols-2 gap-2',
                                      children: [
                                        Field({ label: 'Model', value: form.model, onChange: set('model'), placeholder: 'embeddings' }),
                                        Field({ label: 'Vector dim', type: 'number', value: form.vector_dim, onChange: set('vector_dim'), placeholder: '768' })
                                      ]
                                    }),
                                    Field({ label: 'API key', type: 'password', value: form.api_key, onChange: set('api_key'), placeholder: cfg && cfg.api_key_redacted ? '•••• (current)' : '(none set)' })
                                  ]
                                }),
                                testResult
                                  ? jsxs('div', {
                                      className: 'flex flex-col gap-1 rounded border border-(--ui-stroke-secondary) p-2 text-[0.75rem]',
                                      children: [
                                        jsxs('div', {
                                          className: 'flex items-center gap-2',
                                          children: [
                                            jsx('span', { 'aria-hidden': true, children: testResult.qdrant.ok ? '✅' : '❌' }),
                                            jsx('span', { children: `Qdrant ${testResult.qdrant.ok ? `· ${testResult.qdrant.collections} collection(s)` : `· ${testResult.qdrant.error}` }` })
                                          ]
                                        }),
                                        jsxs('div', {
                                          className: 'flex items-center gap-2',
                                          children: [
                                            jsx('span', { 'aria-hidden': true, children: testResult.embedding.ok ? '✅' : '❌' }),
                                            jsx('span', { children: `Embedding ${testResult.embedding.ok ? `· dim ${testResult.embedding.dim}` : `· ${testResult.embedding.error}` }` })
                                          ]
                                        })
                                      ]
                                    })
                                  : null,
                                jsxs('div', {
                                  className: 'flex items-center justify-end gap-2 border-t border-(--ui-stroke-secondary) pt-2',
                                  children: [
                                    footerBtn('Test', 'ghost', test, busy),
                                    footerBtn('Cancel', 'ghost', () => { setView('status'); setTestResult(null) }, busy),
                                    footerBtn('Save', 'primary', save, busy)
                                  ]
                                })
                              ]
                            })
                      ]
                    })
                  : jsxs('div', {
                      className: 'flex flex-col gap-2 text-[0.8125rem]',
                      children: [
                        jsxs('div', {
                          className: 'flex items-center justify-between gap-2',
                          children: [
                            jsx('div', {
                              className: 'min-w-0 truncate font-medium',
                              title: live ? live.collection : undefined,
                              children: live ? live.collection || 'not indexed' : 'unavailable'
                            }),
                            jsx(
                              'span',
                              {
                                className: cn(
                                  'shrink-0 rounded px-1.5 py-0.5 text-[0.6875rem] tabular-nums',
                                  deleted
                                    ? 'bg-amber-500/15 text-amber-500'
                                    : !live || !indexed || disabled
                                      ? 'bg-muted text-muted-foreground'
                                      : stale
                                        ? 'bg-amber-500/15 text-amber-500'
                                        : 'bg-emerald-500/15 text-emerald-400'
                                ),
                                children: !live ? (transitioning ? 'resolving…' : 'offline') : disabled ? 'disabled' : deleted ? 'deleted' : !indexed ? 'not indexed' : stale ? `stale ${n}~` : 'connected'
                              }
                            )
                          ]
                        }),
                        jsx('div', {
                          className: 'truncate text-[0.75rem] text-muted-foreground',
                          title: cwd,
                          children: cwd
                        }),
                        live
                          ? jsxs('dl', {
                              className: 'grid grid-cols-2 gap-x-3 gap-y-1 text-[0.75rem]',
                              children: [
                                jsxs('div', { className: 'flex justify-between', children: [jsx('dt', { className: 'text-muted-foreground', children: 'Files' }), jsx('dd', { className: 'tabular-nums', children: live.file_count ?? '–' })] }),
                                jsxs('div', { className: 'flex justify-between', children: [jsx('dt', { className: 'text-muted-foreground', children: 'Total' }), jsx('dd', { className: 'tabular-nums', children: live.total ?? '–' })] }),
                                jsxs('div', { className: 'flex justify-between', children: [jsx('dt', { className: 'text-muted-foreground', children: 'Changed' }), jsx('dd', { className: 'tabular-nums text-amber-500', children: live.changed ?? 0 })] }),
                                jsxs('div', { className: 'flex justify-between', children: [jsx('dt', { className: 'text-muted-foreground', children: 'New' }), jsx('dd', { className: 'tabular-nums text-amber-500', children: live.new ?? 0 })] }),
                                lastIndexed
                                  ? jsxs('div', { className: 'col-span-2 flex justify-between', children: [jsx('dt', { className: 'text-muted-foreground', children: 'Last indexed' }), jsx('dd', { children: lastIndexed })] })
                                  : null,
                                live.last_op
                                  ? jsxs('div', { className: 'col-span-2', children: [jsx('dt', { className: 'text-[0.75rem] text-muted-foreground', children: live.last_op.status === 'running' ? `Reindex in progress${opProg ? `… ${opProg} files done` : '…'}` : `Last reindex: ${live.last_op.message || 'done'}` })] })
                                  : null
                              ]
                            })
                          : null,
                        jsxs(
                          'div',
                          {
                            className: 'flex items-center justify-between gap-2 border-t border-(--ui-stroke-secondary) pt-2',
                            children: [
                              footerBtn('⚙️ Settings', 'ghost', () => setView('settings')),
                              jsxs('div', {
                                className: 'flex items-center gap-2',
                                children: [
                                  footerBtn('Close', 'ghost', () => $open.set(false)),
                                  jsx(
                                    'button',
                                    {
                                      type: 'button',
                                      disabled: reindexBusy || transitioning,
                                      onClick: () => {
                                        // Bridge: disable+label immediately, before the
                                        // backend reports the op as running. Cleared on
                                        // the first post-POST refetch, from which point
                                        // data.reindexing / last_op is the source of truth.
                                        setManualBusy(true)
                                        ctx.rest('/reindex?root=' + encodeURIComponent(cwd), { method: 'POST' })
                                          .then(
                                            () => {
                                              host.notify({ kind: 'success', title: 'Qdrant: index started', message: live && live.indexed ? 'Background reindex in progress.' : 'Indexing this project in the background.' })
                                              let tries = 0
                                              const timer = setInterval(() => {
                                                tries += 1
                                                refetch().then(() => {
                                                  if (tries === 1) setManualBusy(false)
                                                  if (tries >= 20) {
                                                    host.notify({ kind: 'info', title: 'Qdrant', message: 'Reindex window ended — check the pill for the latest state.' })
                                                    clearInterval(timer)
                                                  }
                                                })
                                              }, 3000)
                                            },
                                            (err) => {
                                              setManualBusy(false)
                                              host.notifyError(err, 'Qdrant: reindex failed')
                                            }
                                          )
                                      },
                                      className: cn(
                                        'rounded px-2 py-1 text-[0.75rem] font-medium',
                                        'bg-primary text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40'
                                      ),
                                      children: reindexBusy ? 'Indexing…' : (live && live.indexed ? 'Reindex now' : 'Index now')
                                    }
                                  )
                                ]
                              })
                            ]
                          }
                        )
                      ]
                    })
              ]
            }
          )
        ]
      })
    }

    ctx.register({
      id: 'qdrant-pill',
      area: STATUSBAR_AREAS.right,
      order: 90,
      toggleLabel: 'Qdrant Index',
      render: () => jsx(QdrantPill, {})
    })
  }
}
