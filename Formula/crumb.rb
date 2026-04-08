class Crumb < Formula
  include Language::Python::Virtualenv

  desc "Copy-paste AI handoff format"
  homepage "https://github.com/XioAISolutions/crumb-format"
  url "https://files.pythonhosted.org/packages/f3/69/112ad2ca369280c59828ec912420a5fb90f877f1dad0e7b2f7ff21737061/crumb_format-0.2.0.tar.gz"
  sha256 "8fb26a37335541cd00551c98c3af404a47ea7104c5637ac45c3def38900561e9"
  license "MIT"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    crumb = testpath/"sample.crumb"
    crumb.write <<~EOS
      BEGIN CRUMB
      v=1.1
      kind=task
      title=Formula validation smoke test
      source=homebrew.test
      ---
      [goal]
      Verify the Homebrew-installed crumb CLI can validate a simple CRUMB.

      [context]
      - Installed from the Homebrew formula test block

      [constraints]
      - Keep the smoke test deterministic
      END CRUMB
    EOS

    help_output = shell_output("#{bin}/crumb --help")
    assert_match "Create, validate, inspect, and manage .crumb handoff files.", help_output

    output = shell_output("#{bin}/crumb validate #{crumb}")
    assert_match "OK", output
    assert_match "kind=task", output
  end
end
