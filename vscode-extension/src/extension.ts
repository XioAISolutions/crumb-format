import * as vscode from "vscode";
import { exec } from "child_process";

const TASK_TEMPLATE = `--- BEGIN CRUMB ---
v=1
kind=task
title=Untitled Task
source=vscode
target=
created=${new Date().toISOString().split("T")[0]}
status=active
---

[goal]
Describe the task objective here.

[context]
- Relevant background information
- Current state of the system

[steps]
- Step 1
- Step 2
- Step 3

[constraints]
- Any constraints or requirements

[done-when]
- Acceptance criteria here

--- END CRUMB ---
`;

const MEM_TEMPLATE = `--- BEGIN CRUMB ---
v=1
kind=mem
title=Untitled Memory
source=vscode
created=${new Date().toISOString().split("T")[0]}
status=active
---

[summary]
Brief summary of what was learned or decided.

[details]
- Detail 1
- Detail 2

[related]
- Related file or crumb references

--- END CRUMB ---
`;

export function activate(context: vscode.ExtensionContext) {
  // CRUMB: New Task Handoff
  const newTask = vscode.commands.registerCommand("crumb.newTask", async () => {
    const doc = await vscode.workspace.openTextDocument({
      content: TASK_TEMPLATE,
      language: "crumb",
    });
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage("New CRUMB task handoff created.");
  });

  // CRUMB: New Memory Crumb
  const newMem = vscode.commands.registerCommand("crumb.newMem", async () => {
    const doc = await vscode.workspace.openTextDocument({
      content: MEM_TEMPLATE,
      language: "crumb",
    });
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage("New CRUMB memory crumb created.");
  });

  // CRUMB: Validate Current File
  const validate = vscode.commands.registerCommand(
    "crumb.validate",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active file to validate.");
        return;
      }

      const filePath = editor.document.uri.fsPath;
      const terminal = vscode.window.createTerminal("CRUMB Validate");
      terminal.show();
      terminal.sendText(`crumb validate "${filePath}"`);
      vscode.window.showInformationMessage(
        `Running crumb validate on ${filePath}`
      );
    }
  );

  // CRUMB: Compress Current File
  const compress = vscode.commands.registerCommand(
    "crumb.compress",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active file to compress.");
        return;
      }

      const filePath = editor.document.uri.fsPath;
      const terminal = vscode.window.createTerminal("CRUMB Compress");
      terminal.show();
      terminal.sendText(`crumb compress "${filePath}"`);
      vscode.window.showInformationMessage(
        `Running crumb compress on ${filePath}`
      );
    }
  );

  // CRUMB: Bench Current File
  const bench = vscode.commands.registerCommand("crumb.bench", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("No active file to bench.");
      return;
    }

    const filePath = editor.document.uri.fsPath;

    exec(`crumb bench "${filePath}"`, (error, stdout, stderr) => {
      if (error) {
        vscode.window.showErrorMessage(`Bench failed: ${stderr || error.message}`);
        return;
      }

      const outputChannel = vscode.window.createOutputChannel("CRUMB Bench");
      outputChannel.clear();
      outputChannel.appendLine(stdout);
      outputChannel.show();
      vscode.window.showInformationMessage("CRUMB bench complete. See output panel.");
    });
  });

  // CRUMB: Crumb It (generate from selection)
  const crumbIt = vscode.commands.registerCommand(
    "crumb.crumbIt",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("No active editor.");
        return;
      }

      const selection = editor.document.getText(editor.selection);
      if (!selection) {
        vscode.window.showWarningMessage(
          "No text selected. Select text first, then run CRUMB: Crumb It."
        );
        return;
      }

      // Write selection to a temp file, run crumb from-chat, open result
      const tmpFile = vscode.Uri.joinPath(
        context.globalStorageUri,
        "crumb-it-input.txt"
      );
      await vscode.workspace.fs.writeFile(
        tmpFile,
        Buffer.from(selection, "utf-8")
      );

      exec(
        `crumb from-chat "${tmpFile.fsPath}"`,
        (error, stdout, stderr) => {
          if (error) {
            vscode.window.showErrorMessage(
              `Crumb It failed: ${stderr || error.message}`
            );
            return;
          }

          vscode.workspace
            .openTextDocument({ content: stdout, language: "crumb" })
            .then((doc) => {
              vscode.window.showTextDocument(doc);
              vscode.window.showInformationMessage(
                "CRUMB generated from selection."
              );
            });
        }
      );
    }
  );

  context.subscriptions.push(newTask, newMem, validate, compress, bench, crumbIt);
}

export function deactivate() {}
