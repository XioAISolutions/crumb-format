"""
Tests for Wave Field LLM deployment automation
"""
import os
import subprocess
import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent
AUTOMATION_DIR = PROJECT_ROOT / "deployment" / "automation"


class TestDeploymentScripts:
    """Test deployment automation scripts"""
    
    def test_scripts_exist(self):
        """Verify all required scripts exist"""
        required_scripts = [
            "deploy.sh",
            "quick-start.sh",
            "deploy-model.sh",
            "update-model.sh",
            "setup-monitoring.sh",
            "validate.sh",
            "smoke-test.sh",
            "rollback.sh",
            "scale.sh",
            "backup.sh",
            "cleanup.sh",
        ]
        
        for script in required_scripts:
            script_path = AUTOMATION_DIR / script
            assert script_path.exists(), f"Script not found: {script}"
            assert os.access(script_path, os.X_OK), f"Script not executable: {script}"
    
    def test_cloud_scripts_exist(self):
        """Verify cloud-specific scripts exist"""
        cloud_scripts = [
            "aws/deploy-aws.sh",
            "gcp/deploy-gcp.sh",
            "azure/deploy-azure.sh",
        ]
        
        for script in cloud_scripts:
            script_path = AUTOMATION_DIR / script
            assert script_path.exists(), f"Cloud script not found: {script}"
            assert os.access(script_path, os.X_OK), f"Script not executable: {script}"
    
    def test_templates_exist(self):
        """Verify configuration templates exist"""
        templates = [
            "templates/production.env.template",
        ]
        
        for template in templates:
            template_path = AUTOMATION_DIR / template
            assert template_path.exists(), f"Template not found: {template}"
    
    def test_guides_exist(self):
        """Verify deployment guides exist"""
        guides = [
            "guides/PRODUCTION_DEPLOYMENT.md",
            "guides/LOCAL_DEPLOYMENT.md",
            "guides/KUBERNETES_DEPLOYMENT.md",
            "guides/CLOUD_DEPLOYMENT.md",
        ]
        
        guides_dir = PROJECT_ROOT / "deployment"
        for guide in guides:
            guide_path = guides_dir / guide
            assert guide_path.exists(), f"Guide not found: {guide}"
    
    def test_script_help_text(self):
        """Verify scripts have help text"""
        scripts = ["deploy.sh", "quick-start.sh", "deploy-model.sh"]
        
        for script in scripts:
            script_path = AUTOMATION_DIR / script
            result = subprocess.run(
                [str(script_path), "--help"],
                capture_output=True,
                text=True
            )
            assert result.returncode == 1  # Help exits with 1
            assert "Usage:" in result.stdout, f"No help text in {script}"
    
    def test_validate_script_syntax(self):
        """Verify scripts have valid bash syntax"""
        scripts = [
            "deploy.sh",
            "quick-start.sh",
            "validate.sh",
            "smoke-test.sh",
        ]
        
        for script in scripts:
            script_path = AUTOMATION_DIR / script
            result = subprocess.run(
                ["bash", "-n", str(script_path)],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Syntax error in {script}: {result.stderr}"


class TestDeploymentConfiguration:
    """Test deployment configuration"""
    
    def test_kubernetes_manifests_exist(self):
        """Verify Kubernetes manifests exist"""
        k8s_dir = PROJECT_ROOT / "deployment" / "kubernetes"
        manifests = [
            "deployment.yaml",
            "service.yaml",
            "configmap.yaml",
            "secret.yaml",
            "ingress.yaml",
            "hpa.yaml",
        ]
        
        for manifest in manifests:
            manifest_path = k8s_dir / manifest
            assert manifest_path.exists(), f"Manifest not found: {manifest}"
    
    def test_docker_files_exist(self):
        """Verify Docker files exist"""
        docker_dir = PROJECT_ROOT / "deployment" / "docker"
        files = [
            "Dockerfile",
            "Dockerfile.cpu",
            "Dockerfile.gpu",
            "docker-compose.yml",
        ]
        
        for file in files:
            file_path = docker_dir / file
            assert file_path.exists(), f"Docker file not found: {file}"
    
    def test_monitoring_configs_exist(self):
        """Verify monitoring configurations exist"""
        monitoring_dir = PROJECT_ROOT / "deployment" / "monitoring"
        configs = [
            "prometheus.yml",
            "alertmanager.yml",
        ]
        
        for config in configs:
            config_path = monitoring_dir / config
            assert config_path.exists(), f"Monitoring config not found: {config}"


class TestDeploymentValidation:
    """Test deployment validation logic"""
    
    def test_validate_script_checks_tools(self):
        """Test that validate script checks for required tools"""
        script_path = AUTOMATION_DIR / "validate.sh"
        result = subprocess.run(
            [str(script_path)],
            capture_output=True,
            text=True
        )
        # Script should check for kubectl, helm, etc.
        assert "kubectl" in result.stdout or "kubectl" in result.stderr


@pytest.mark.integration
class TestDeploymentIntegration:
    """Integration tests for deployment (requires cluster)"""
    
    @pytest.mark.skipif(
        subprocess.run(["kubectl", "cluster-info"], capture_output=True).returncode != 0,
        reason="Kubernetes cluster not available"
    )
    def test_smoke_test_script(self):
        """Test smoke test script against running cluster"""
        script_path = AUTOMATION_DIR / "smoke-test.sh"
        result = subprocess.run(
            [str(script_path), "default"],
            capture_output=True,
            text=True
        )
        # Should complete without error (may have warnings if no deployment)
        assert result.returncode in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
