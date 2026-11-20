{{- define "modelrouter.fullname" -}}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- define "modelrouter.labels" -}}
app.kubernetes.io/name: modelrouter
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}
{{- define "modelrouter.selector" -}}
app.kubernetes.io/name: modelrouter
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
