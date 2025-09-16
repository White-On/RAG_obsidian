"""
Script pour lancer l'application Streamlit RAG
"""
import subprocess
import sys
from pathlib import Path

def run_streamlit_app():
    """Lance l'application Streamlit"""
    app_path = Path(__file__).parent / "app.py"
    
    try:
        # Commande pour lancer streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        
        print("🚀 Lancement de l'application Streamlit RAG...")
        print(f"📁 Fichier: {app_path}")
        print("🌐 L'application sera accessible à: http://localhost:8501")
        print("🛑 Appuyez sur Ctrl+C pour arrêter l'application")
        print("-" * 50)
        
        # Lancer l'application
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    run_streamlit_app()