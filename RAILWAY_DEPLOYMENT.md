# Documentation technique — Déploiement Railway du backend Attentiq

## 1. Création du projet Railway depuis le repo GitHub

1. Se connecter sur [railway.app](https://railway.app).
2. Cliquer **New Project → Deploy from GitHub repo**.
3. Autoriser Railway à accéder au repo `nanocorp-hq/attentiq` si ce n'est pas déjà fait.
4. Sélectionner le repo. Railway liste les branches disponibles ; sélectionner `main` (commit cible : `2fc34ff`).

> Railway déploie automatiquement depuis la branche sélectionnée. Pour pointer sur un commit précis, créer une branche/tag depuis ce commit avant de connecter le repo.

---

## 2. Configuration du dossier racine (Root Directory)

Dans les **Settings** du service Railway → section **Source** :

```
Root Directory: backend/
```

**Pourquoi c'est nécessaire** : le `Dockerfile` et `main.py` se trouvent dans `backend/`, pas à la racine du repo. Sans ce paramètre, Railway cherche le `Dockerfile` à la racine et échoue à builder.

---

## 3. Variables d'environnement requises

Dans **Settings → Variables** du service Railway :

| Variable | Valeur | Source |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` (clé OpenAI de l'utilisateur) | **Saisir manuellement** |
| `PORT` | `8000` | **Injectée automatiquement par Railway** |

**Détail :**
- `OPENAI_API_KEY` : utilisée pour Whisper (`whisper-1`) et GPT-4o. Doit avoir accès aux deux modèles. À saisir manuellement dans le dashboard Railway.
- `PORT` : Railway injecte cette variable automatiquement à chaque déploiement. Le Dockerfile définit `ENV PORT=8000` comme valeur par défaut ; la valeur injectée par Railway prend le dessus. Aucune saisie manuelle requise.

---

## 4. Commande de démarrage et port exposé

Railway détecte la commande de démarrage via le `CMD` du Dockerfile :

```dockerfile
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 130"]
```

- **Port exposé** : `8000` (valeur par défaut de `$PORT`).
- **Procfile** : non nécessaire. Le `Dockerfile` et le `railway.json` présents dans `backend/` sont suffisants ; Railway les utilise directement.
- `railway.json` configure également `healthcheckPath: /health` avec un timeout de 30 secondes.

---

## 5. Vérification du endpoint `/health`

Une fois le déploiement terminé, Railway génère une URL publique au format :

```
https://[service-name].up.railway.app
```

**Commande de vérification :**

```bash
curl https://[service-name].up.railway.app/health
```

**Réponse attendue :**

```json
{"status": "ok", "service": "attentiq-backend"}
```

**Si la réponse est 502 ou 503 :**
- Aller dans Railway → onglet **Deployments** → cliquer sur le déploiement actif → **View Logs**.
- Vérifier que uvicorn démarre sans erreur et que `OPENAI_API_KEY` est bien définie.
- Un 503 immédiat indique souvent que le healthcheck échoue pendant le démarrage ; attendre 30 secondes supplémentaires puis retester.

---

## 6. Test du endpoint `POST /analyze` avec une URL réelle

**Commande curl :**

```bash
curl -X POST https://[service-name].up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "url": "https://www.tiktok.com/@saleswitheva/video/7234567890123456789",
    "platform": "tiktok",
    "max_duration_seconds": 60,
    "requested_at": "2026-04-18T10:00:00Z"
  }'
```

**Headers requis :**
- `Content-Type: application/json`

**Réponse attendue (`status: "success"`) :**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "metadata": {
    "url": "...",
    "platform": "tiktok",
    "author": "...",
    "title": "...",
    "duration_seconds": 45.0,
    "hashtags": ["#sales", "..."]
  },
  "transcript": [
    { "start": 0.0, "end": 3.2, "text": "..." }
  ],
  "visual_signals": [
    {
      "timestamp_seconds": 0,
      "face_expression": "...",
      "body_position": "...",
      "on_screen_text": "...",
      "motion_level": "medium",
      "scene_change": false
    }
  ],
  "diagnostic": {
    "retention_score": 6.5,
    "global_summary": "...",
    "drop_off_rule": "...",
    "creator_perception": "...",
    "attention_drops": [
      { "timestamp_seconds": 12, "severity": "medium", "cause": "..." }
    ],
    "audience_loss_estimate": "...",
    "corrective_actions": ["...", "...", "..."]
  },
  "processing_time_seconds": 47.3
}
```

**Temps de réponse attendu** : inférieur à 120 secondes. Le pipeline télécharge la vidéo, extrait l'audio, transcrit via Whisper, analyse les frames via GPT-4o Vision, puis génère le diagnostic GPT-4o — comptez 30 à 90 secondes selon la durée de la vidéo.

---

## 7. Erreurs possibles et résolution

| Code d'erreur | HTTP | Cause probable | Action |
|---|---|---|---|
| `VIDEO_UNAVAILABLE` | 404 | Vidéo supprimée, privée, ou URL invalide | Vérifier que l'URL est accessible publiquement dans un navigateur |
| `DURATION_EXCEEDED` | 400 | Vidéo plus longue que `max_duration_seconds` (défaut : 60s) | Réduire à une vidéo ≤ 60s, ou augmenter `max_duration_seconds` dans la requête |
| `TRANSCRIPT_FAILED` | 200 (`status: "partial"`) | Échec de l'extraction audio ou de l'appel Whisper | La réponse est retournée sans transcript ; vérifier `OPENAI_API_KEY` et le quota Whisper |
| `VISION_FAILED` | 200 (`status: "partial"`) | Échec de l'analyse GPT-4o Vision sur les frames | La réponse est retournée sans `visual_signals` ; vérifier que la clé OpenAI a accès à GPT-4o |
| Erreur 500 générique | 500 | Erreur inattendue dans le pipeline | Consulter **Railway → Deployments → View Logs** pour le message complet |
| Timeout > 120s | 504 | Pipeline trop long (vidéo longue, latence OpenAI) | Réessayer avec une vidéo plus courte ; si récurrent, vérifier les logs Railway pour identifier l'étape bloquante |
