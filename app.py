"""
Application Streamlit pour le système RAG
"""

import streamlit as st
from pathlib import Path

from src.rag_system import RAGSystem
import src.config as config

# Configuration de la page
st.set_page_config(
    page_title="RAG System - Question Answering", page_icon="🤖", layout="wide"
)


def initialize_rag_system():
    """Initialise le système RAG et le met en cache"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.initialized = False

    return st.session_state.rag_system


def display_documents_info(docs_info):
    """Affiche les informations sur les documents"""
    st.subheader("📚 Informations sur les Documents")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Documents Chargés", docs_info.get("total_documents", 0))

    with col2:
        st.metric("Chunks Créés", docs_info.get("total_chunks", 0))

    with col3:
        status = docs_info.get("status", "Unknown")
        if status == "Documents loaded successfully":
            st.success("✅ " + status)
        elif status == "No documents loaded":
            st.warning("⚠️ " + status)
        else:
            st.info("ℹ️ " + status)

    # Affichage des fichiers chargés
    if docs_info.get("file_paths"):
        st.subheader("📄 Fichiers Chargés")
        with st.expander("Voir la liste des fichiers", expanded=False):
            for i, file_path in enumerate(docs_info["file_paths"], 1):
                file_name = Path(file_path).name
                st.write(f"{i}. `{file_name}`")
                with st.expander(f"Chemin complet - {file_name}", expanded=False):
                    st.code(file_path)


def main():
    """Fonction principale de l'application Streamlit"""

    # Titre principal
    st.title("🤖 Système RAG - Question Answering")
    st.markdown("---")

    # Initialisation du système RAG
    rag_system = initialize_rag_system()

    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Paramètres de chargement
        st.subheader("Chargement des Documents")
        doc_limit = st.number_input(
            "Limite de documents",
            min_value=1,
            max_value=100,
            value=10,
            help="Nombre maximum de documents à charger",
        )

        # Paramètres de chunking
        st.subheader("Paramètres de Chunking")
        chunk_size = st.slider("Taille des chunks", 1000, 10000, 5000, 500)
        chunk_overlap = st.slider("Chevauchement", 0, 1000, 200, 50)

        # Bouton d'initialisation
        if st.button("🚀 Initialiser le Système RAG", type="primary"):
            with st.spinner("Initialisation en cours..."):
                def progress_callback(batch_num, total_batches):
                    st.write(f"📊 Traitement du batch {batch_num}/{total_batches}")
                
                result = rag_system.initialize(limit=doc_limit, progress_callback=progress_callback)
                
                if result["status"] == "success":
                    st.session_state.initialized = True
                    st.success("✅ Système initialisé avec succès!")
                    st.session_state.docs_info = result["documents_info"]
                    
                    # Afficher les statistiques d'embedding si disponibles
                    if "embedding_stats" in result:
                        st.info(f"🔧 Embedding: {result['embedding_stats']['success_rate']:.1%} de succès, "
                               f"{result['embedding_stats']['total_processing_time']:.1f}s, "
                               f"{result['embedding_stats']['cache_hits']} cache hits")
                else:
                    st.error(f"❌ Erreur: {result['message']}")        # Affichage du statut
        if hasattr(st.session_state, "initialized") and st.session_state.initialized:
            st.success("🟢 Système prêt")
        else:
            st.warning("🟡 Système non initialisé")

    # Interface principale
    if hasattr(st.session_state, "initialized") and st.session_state.initialized:
        # Affichage des informations sur les documents
        if hasattr(st.session_state, "docs_info"):
            display_documents_info(st.session_state.docs_info)
            st.markdown("---")

        # Section de questions-réponses
        st.subheader("💬 Posez vos Questions")

        # Zone de saisie de la question
        question = st.text_area(
            "Votre question:", placeholder="Tapez votre question ici...", height=100
        )

        # Bouton pour poser la question
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Poser la Question", type="primary")

        # Traitement de la question
        if ask_button and question.strip():
            with st.spinner("Recherche en cours..."):
                try:
                    answer = rag_system.ask_question(question)

                    # Affichage de la réponse
                    st.subheader("🎯 Réponse")
                    st.info(answer)

                    # Historique des questions (optionnel)
                    if "qa_history" not in st.session_state:
                        st.session_state.qa_history = []

                    st.session_state.qa_history.append(
                        {"question": question, "answer": answer}
                    )

                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement de la question: {str(e)}")

        elif ask_button and not question.strip():
            st.warning("⚠️ Veuillez saisir une question.")

        # Affichage de l'historique
        if hasattr(st.session_state, "qa_history") and st.session_state.qa_history:
            st.markdown("---")
            st.subheader("📝 Historique des Questions")

            with st.expander("Voir l'historique", expanded=False):
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:]), 1):
                    st.write(f"**Q{i}:** {qa['question']}")
                    st.write(f"**R{i}:** {qa['answer']}")
                    st.markdown("---")

    else:
        # Instructions si le système n'est pas initialisé
        st.info("👈 Utilisez la barre latérale pour initialiser le système RAG")

        st.subheader("📖 Comment utiliser cette application")
        st.markdown("""
        1. **Configurez les paramètres** dans la barre latérale
        2. **Cliquez sur "Initialiser le Système RAG"** pour charger les documents
        3. **Attendez** que l'initialisation se termine
        4. **Posez vos questions** dans la zone de texte principale
        5. **Consultez les réponses** générées par le système
        """)

        st.subheader("🔧 Configuration Actuelle")
        st.write(f"**Répertoire des documents:** `{config.FILE_VAULT}`")
        st.write(f"**Base de données Chroma:** `{config.CHROMA_DB_DIR}`")


if __name__ == "__main__":
    main()
