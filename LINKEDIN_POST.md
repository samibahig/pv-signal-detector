# LinkedIn Post — Pharmacovigilance Signal Detector

---

🔴 Ce graphique a peut-être sauvé des vies. Voici pourquoi il n'est pas un simple scatter plot.

---

J'ai construit un outil de détection de signaux de pharmacovigilance avec Plotly et Dash — et je voulais expliquer pourquoi ce type de visualisation est fondamentalement différent d'un graphique classique, et pourquoi il manquait à l'écosystème Python.

---

**Un scatter plot standard montre une corrélation.**
Ce graphique, lui, encode une décision réglementaire.

Chaque point représente une paire (médicament, effet indésirable). Derrière chaque point se cachent quatre chiffres bruts — les cellules d'un tableau de contingence — et une question qui se pose dans tous les départements de pharmacovigilance du monde :

> *Est-ce que ce médicament provoque cet effet indésirable plus souvent qu'on ne s'y attendrait par hasard ?*

---

**Ce que le graphique calcule automatiquement :**

📐 **PRR** (Proportional Reporting Ratio) — le signal de base. Un PRR ≥ 2 déclenche une alerte dans la plupart des agences réglementaires (FDA, EMA).

📐 **ROR** (Reporting Odds Ratio) + intervalle de confiance à 95 % — robustesse statistique du signal.

📐 **IC** (Information Component) — mesure bayésienne issue du BCPNN : combien le médicament et l'effet coexistent au-delà du hasard, en bits d'information.

📐 **EBGM** (Empirical Bayes Geometric Mean) — estimateur bayésien qui compresse les signaux instables sur les événements rares. Utilisé historiquement par la FDA dans leurs bases FAERS.

---

**Pourquoi c'est différent d'un scatter plot ?**

Un scatter classique vous montre où sont les points. Ce graphique vous dit quoi faire avec chaque point :

🔴 PRR élevé + n élevé → **signal prioritaire** à investiguer immédiatement
🟡 PRR élevé + n faible → **signal rare** à valider (potentiel mais fragile)
🟠 PRR faible + n élevé → **effet fréquent mais non spécifique** (bruit de fond connu)
🟢 PRR faible + n faible → **bruit** — à ignorer

Les lignes en pointillés ne sont pas esthétiques. Ce sont des seuils réglementaires (PRR = 2, n = 3) utilisés dans les workflows réels des comités de pharmacovigilance.

---

**Pourquoi ce n'est pas encore dans Plotly ?**

Plotly Express est conçu pour être généraliste. `px.scatter()`, `px.histogram()`, `px.box()` — des primitives universelles.

`px.signal_detection()` n'existe pas encore. Ce projet implémente exactement ce que cette fonction devrait faire : ingérer un tableau de contingence brut, calculer tous les métriques, appliquer les règles réglementaires, et produire une visualisation prête à décision — en un seul appel.

C'est une contribution que je souhaite proposer à la communauté Plotly, et plus largement aux data scientists qui travaillent en pharmacovigilance et reproduisent ce workflow manuellement des centaines de fois.

---

**Ce que l'outil permet aujourd'hui :**

✅ Chargement d'un CSV brut (colonnes : drug, event, a, b, c, d)
✅ Calcul automatique de PRR, ROR, IC, EBGM avec smoothing de Laplace
✅ Classification signal / watch / background selon les règles FDA/EMA
✅ Quadrant de décision interactif avec seuils ajustables en temps réel
✅ Table des signaux classée par score de priorité
✅ Compatible FAERS, EudraVigilance, bases hospitalières
✅ Déployé sur Hugging Face Spaces (Docker) — accessible à tous

---

**Lien Hugging Face Space :**
👉 https://huggingface.co/spaces/samibahig-md/pv-signal-detector

**Code source GitHub :**
👉 https://github.com/samibahig/pv-signal-detector

---

Si vous travaillez en pharmacovigilance, en sécurité des médicaments, ou si vous êtes data scientist dans l'industrie pharmaceutique — je serais curieux d'avoir votre retour.

Est-ce que ce type d'outil vous serait utile dans votre workflow quotidien ?

---

#Pharmacovigilance #DrugSafety #Plotly #DataScience #Python #OpenSource #MedicalData #PharmaData #FAERS #SignalDetection #HuggingFace #HealthcareAI
