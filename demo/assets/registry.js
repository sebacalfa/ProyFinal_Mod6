/* Evidencia de la sesión actual: no asignar estados de aprobación estáticos. */
const evidenceText = value => String(value ?? 'No aplica').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const evidenceNumber = value => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('es-CR', {maximumFractionDigits: 5}) : 'No aplica';
const evidenceStatus = value => ({completed:'Completado',selected:'Seleccionado',running:'En revisión',passed:'Aprobado',approved:'Aprobado',rejected:'Rechazado',error:'Error'}[value] || value || 'Sin evidencia');
const evidenceStage = value => ({Experiment:'Experimento',Candidate:'Candidato',Validation:'Validación',Production:'Producción'}[value] || value);

function renderRegistryProof(evidence, active) {
  if (!evidence) return '<article class="card registry-proof"><h2>El servidor necesita reiniciarse</h2><p>La pantalla está actualizada, pero este servidor todavía no envía la información de Registry. Esto no significa que se hayan perdido los registros.</p><a class="external-action" href="http://127.0.0.1:8010">Abrir la demo actualizada ↗</a></article>';
  const registry = evidence?.lifecycle;
  if (!registry?.version) return '<article class="card registry-proof"><h2>Registro del modelo</h2><p>No hay un historial de aprobación disponible para esta ejecución.</p></article>';
  const events = registry.events || [], aliases = evidence.aliases || [];
  const production = aliases.find(item => item.alias === 'production');
  const approved = registry.stage === 'Production' && registry.status === 'approved' && production?.version === String(registry.version);
  const serving = active?.loaded && String(active.version) === String(registry.version);
  const names = {cv_average_precision:'Average Precision en CV',cv_f1:'F1 en CV (umbral 0,5)',gain_over_baseline:'Mejora sobre el modelo básico',cv_average_precision_std:'Desviación estándar de Average Precision'};
  const checks = Object.entries(registry.decision?.checks || {}).map(([key, check]) => `<tr><th scope="row">${evidenceText(names[key] || key)}</th><td>${evidenceNumber(check.actual)}</td><td>${check.minimum != null ? '≥ '+evidenceNumber(check.minimum) : check.maximum != null ? '≤ '+evidenceNumber(check.maximum) : 'No aplica'}</td><td><span class="badge ${check.passed ? 'selected' : 'warn'}">${check.passed ? 'Cumple' : 'No cumple'}</span></td></tr>`).join('');
  const history = events.map(event => {
    const date = new Date(event.at_utc);
    const formatted = Number.isNaN(date.getTime()) ? 'No aplica' : date.toLocaleString('es-CR', {timeZone:'America/Costa_Rica',hour12:false});
    return `<tr><th scope="row">${evidenceText(evidenceStage(event.stage))}</th><td>${evidenceText(evidenceStatus(event.status))}</td><td>${evidenceText(formatted)}</td></tr>`;
  }).join('');
  const steps = ['Experiment','Candidate','Validation','Production'].map((stage, i) => {
    const last = [...events].reverse().find(event => event.stage === stage);
    return `${i ? '<i aria-hidden="true">→</i>' : ''}<div class="${last?.status === 'approved' ? 'production' : ''}"><b>${i+1}</b><span>${evidenceText(evidenceStage(stage))}</span><small>${evidenceText(evidenceStatus(last?.status))}</small></div>`;
  }).join('');
  return `<article class="card registry-proof" id="registryEvidence"><div class="card-head"><div><span class="eyebrow">K · REGISTRO DEL MODELO</span><h2>Versión ${evidenceText(registry.version)} · ${approved ? 'Aprobada para producción local' : evidenceText(evidenceStatus(registry.status))}</h2></div><span class="badge ${approved ? 'selected' : 'warn'}">${evidenceText(registry.stage)}</span></div>
    ${evidence.error ? `<p class="evidence-notice">${evidenceText(evidence.error)}</p>` : ''}
    <div class="registry-steps">${steps}</div>
    <div class="trace-grid"><div><small>Modelo registrado</small><strong>${evidenceText(registry.model_name)}</strong></div><div><small>Modelo cargado por la demo</small><strong>${active?.loaded ? 'Versión '+evidenceText(active.version) : 'No disponible'}</strong></div><div><small>Conexión con la API</small><strong>${serving ? 'Coincide con esta versión' : 'Revisar o reiniciar el servidor'}</strong></div><div><small>Política académica</small><strong>${evidenceText(registry.policy_version)}</strong></div></div>
    <div class="artifact-list">${aliases.map(item=>`<span>${evidenceText(item.alias)} → v${evidenceText(item.version)}</span>`).join('') || '<span>Sin alias disponibles</span>'}</div>
    <h3>Criterios de aprobación</h3><p class="plain-note">Se elige la mayor Average Precision promedio en validación cruzada. Estos límites son la política académica del proyecto. El test se informa por separado y no decide la promoción.</p>
    <div class="evidence-scroll"><table class="evidence-table"><thead><tr><th>Criterio</th><th>Resultado</th><th>Regla</th><th>Estado</th></tr></thead><tbody>${checks}</tbody></table></div>
    <p class="plain-note">Carga desde Registry y comparación de predicciones: <strong>${registry.registry_roundtrip_passed ? 'Verificadas' : 'Sin confirmación'}</strong>. La API usa la exportación aprobada.</p>
    <details class="evidence-history" open><summary>Historial de decisiones · hora de Costa Rica</summary><div class="evidence-scroll"><table class="evidence-table"><thead><tr><th>Etapa</th><th>Estado</th><th>Fecha y hora</th></tr></thead><tbody>${history}</tbody></table></div></details></article>`;
}

function renderTrackingProof(evidence) {
  if (!evidence) return '<article class="card tracking-evidence"><span class="eyebrow">J · EXPERIMENTOS Y ARTEFACTOS</span><h2>Servidor anterior detectado</h2><p>Este servidor no envía la evidencia de J y K. Reinícialo para cargar el código actualizado o abre la demo actualizada. No se ha comprobado que falten experimentos.</p><a class="external-action" href="http://127.0.0.1:8010">Abrir la demo actualizada ↗</a></article>';
  const runs = evidence?.runs || [], complete = runs.filter(run => run.complete && run.status === 'FINISHED').length;
  const names = {hist_gradient_boosting:'HistGradientBoosting',random_forest:'Random Forest',logistic_regression:'Regresión logística',dummy_baseline:'Modelo básico',selected_model:'Modelo seleccionado'};
  const rows = runs.map(run => `<tr><th scope="row">${evidenceText(names[run.role] || run.role)}<small>${evidenceText(run.evaluation)}</small></th><td>${run.parameters_complete ? 'Completos' : 'Faltan parámetros'}<small>${evidenceText(run.metrics_count)} métricas registradas</small></td><td><div class="evidence-files">${Object.entries(run.artifacts).map(([name, available])=>`<span class="${available ? 'available' : 'missing'}">${available ? '✓' : '!'} ${evidenceText(name)}${available ? '' : ' · no disponible'}</span>`).join('')}</div></td><td><a class="external-action" href="http://127.0.0.1:5000/#/experiments/${encodeURIComponent(run.experiment_id)}/runs/${encodeURIComponent(run.run_id)}" target="_blank" rel="noopener">Ver run ↗</a></td></tr>`).join('');
  return `<article class="card tracking-evidence" id="trackingEvidence"><div class="card-head"><div><span class="eyebrow">J · EXPERIMENTOS Y ARTEFACTOS</span><h2>Evidencia de cada ejecución</h2></div><span class="badge ${runs.length && complete === runs.length ? 'selected' : 'warn'}">${complete} de ${runs.length} completos</span></div><p class="plain-note">Disponibilidad comprobada en la base de MLflow y en los archivos locales. Cada candidato tiene sus propios gráficos y matriz con predicciones de validación cruzada; el seleccionado tiene la evaluación final en test.</p>${evidence?.error ? `<p class="evidence-notice">${evidenceText(evidence.error)}</p>` : ''}<div class="evidence-scroll"><table class="evidence-table"><thead><tr><th>Ejecución</th><th>Parámetros y métricas</th><th>Artefactos disponibles</th><th>Evidencia</th></tr></thead><tbody>${rows || '<tr><td colspan="4">No hay evidencia disponible.</td></tr>'}</tbody></table></div></article>`;
}
