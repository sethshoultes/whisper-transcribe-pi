const { dialog, shell, app } = require('electron');
const fs = require('fs');
const path = require('path');

class ErrorHandler {
  constructor(logFunction) {
    this.log = logFunction || console.log;
  }

  // Show a comprehensive error dialog with recovery options
  async showErrorWithRecovery(error, context = 'Unknown') {
    const errorType = this.categorizeError(error);
    
    let message, detail, buttons, actions;
    
    switch (errorType) {
      case 'PYTHON_NOT_FOUND':
        ({ message, detail, buttons, actions } = this.getPythonNotFoundDialog());
        break;
      case 'DEPENDENCIES_MISSING':
        ({ message, detail, buttons, actions } = this.getDependenciesMissingDialog(error));
        break;
      case 'PERMISSION_DENIED':
        ({ message, detail, buttons, actions } = this.getPermissionDeniedDialog());
        break;
      case 'DISK_SPACE':
        ({ message, detail, buttons, actions } = this.getDiskSpaceDialog());
        break;
      case 'NETWORK_ERROR':
        ({ message, detail, buttons, actions } = this.getNetworkErrorDialog());
        break;
      default:
        ({ message, detail, buttons, actions } = this.getGenericErrorDialog(error, context));
    }

    const result = await dialog.showMessageBox({
      type: 'error',
      title: 'Whisper Transcribe Error',
      message,
      detail,
      buttons,
      defaultId: 0,
      cancelId: buttons.length - 1
    });

    // Execute the selected action
    if (actions[result.response]) {
      await actions[result.response]();
    }

    return result.response;
  }

  categorizeError(error) {
    const errorMessage = error.message.toLowerCase();
    
    if (errorMessage.includes('python') && (errorMessage.includes('not found') || errorMessage.includes('enoent'))) {
      return 'PYTHON_NOT_FOUND';
    }
    if (errorMessage.includes('modulenotfounderror') || errorMessage.includes('no module named')) {
      return 'DEPENDENCIES_MISSING';
    }
    if (errorMessage.includes('permission denied') || errorMessage.includes('eacces')) {
      return 'PERMISSION_DENIED';
    }
    if (errorMessage.includes('enospc') || errorMessage.includes('disk') || errorMessage.includes('space')) {
      return 'DISK_SPACE';
    }
    if (errorMessage.includes('network') || errorMessage.includes('connection') || errorMessage.includes('timeout')) {
      return 'NETWORK_ERROR';
    }
    
    return 'GENERIC';
  }

  getPythonNotFoundDialog() {
    return {
      message: 'Python Not Found',
      detail: `Whisper Transcribe requires Python 3.8 or higher, but no suitable Python installation was found.

To fix this issue:
1. Install Python from python.org or using Homebrew
2. Ensure Python is accessible from the command line
3. Restart Whisper Transcribe

For macOS users:
• Install via Homebrew: brew install python@3.11
• Or download from: https://www.python.org/downloads/`,
      buttons: ['Open Python Website', 'Show Installation Guide', 'Try Again', 'Quit'],
      actions: [
        () => shell.openExternal('https://www.python.org/downloads/'),
        () => this.showInstallationGuide(),
        () => app.relaunch(),
        () => app.quit()
      ]
    };
  }

  getDependenciesMissingDialog(error) {
    const missingModule = this.extractMissingModule(error.message);
    
    return {
      message: 'Python Dependencies Missing',
      detail: `Required Python packages are not installed.
${missingModule ? `Missing: ${missingModule}` : ''}

Whisper Transcribe will attempt to install the required packages automatically.

If automatic installation fails, you can install manually using:
pip3 install --user flask flask-cors openai-whisper scipy numpy psutil sounddevice`,
      buttons: ['Auto Install', 'Manual Instructions', 'Try Again', 'Quit'],
      actions: [
        () => this.attemptAutoInstall(),
        () => this.showManualInstallInstructions(),
        () => app.relaunch(),
        () => app.quit()
      ]
    };
  }

  getPermissionDeniedDialog() {
    return {
      message: 'Permission Denied',
      detail: `Whisper Transcribe doesn't have the necessary permissions to install or access Python packages.

To fix this:
1. Run the installation with appropriate permissions
2. Or install packages to your user directory:
   pip3 install --user [package-name]
3. Check file and folder permissions

For macOS: You may need to grant additional permissions in System Preferences → Security & Privacy.`,
      buttons: ['Grant Permissions', 'Install to User Directory', 'Help', 'Quit'],
      actions: [
        () => this.requestPermissions(),
        () => this.showUserInstallInstructions(),
        () => this.showPermissionHelp(),
        () => app.quit()
      ]
    };
  }

  getDiskSpaceDialog() {
    return {
      message: 'Insufficient Disk Space',
      detail: `There isn't enough free disk space to install the required components.

Whisper AI models and dependencies require approximately:
• 500MB for the Whisper model
• 200MB for Python packages
• 100MB for temporary files

Please free up at least 1GB of disk space and try again.`,
      buttons: ['Open Storage Settings', 'Clean Temporary Files', 'Try Again', 'Quit'],
      actions: [
        () => this.openStorageSettings(),
        () => this.cleanTemporaryFiles(),
        () => app.relaunch(),
        () => app.quit()
      ]
    };
  }

  getNetworkErrorDialog() {
    return {
      message: 'Network Connection Error',
      detail: `Unable to download required components due to network issues.

This could be caused by:
• No internet connection
• Firewall blocking the download
• Server temporarily unavailable

Please check your internet connection and try again.`,
      buttons: ['Check Connection', 'Retry Download', 'Offline Help', 'Quit'],
      actions: [
        () => shell.openExternal('https://www.google.com'),
        () => app.relaunch(),
        () => this.showOfflineHelp(),
        () => app.quit()
      ]
    };
  }

  getGenericErrorDialog(error, context) {
    return {
      message: `Error in ${context}`,
      detail: `An unexpected error occurred:

${error.message}

This error has been logged for debugging. You can:
1. Try restarting the application
2. Check the log file for more details
3. Report this issue if it persists`,
      buttons: ['Restart App', 'Show Logs', 'Report Issue', 'Quit'],
      actions: [
        () => app.relaunch(),
        () => this.showLogFile(),
        () => this.reportIssue(error, context),
        () => app.quit()
      ]
    };
  }

  extractMissingModule(errorMessage) {
    const match = errorMessage.match(/No module named '([^']+)'/);
    return match ? match[1] : null;
  }

  async showInstallationGuide() {
    const guideText = `# Python Installation Guide for Whisper Transcribe

## macOS Installation

### Option 1: Homebrew (Recommended)
1. Install Homebrew if you don't have it: https://brew.sh
2. Install Python: \`brew install python@3.11\`
3. Restart Whisper Transcribe

### Option 2: Official Python Installer
1. Download Python from: https://www.python.org/downloads/
2. Run the installer
3. Make sure to check "Add Python to PATH"
4. Restart your computer
5. Restart Whisper Transcribe

## Verify Installation
Open Terminal and run: \`python3 --version\`
You should see Python 3.8 or higher.

## Still Having Issues?
Contact support or visit our documentation.`;

    await dialog.showMessageBox({
      type: 'info',
      title: 'Installation Guide',
      message: 'Python Installation Guide',
      detail: guideText,
      buttons: ['OK']
    });
  }

  async attemptAutoInstall() {
    // This would trigger the dependency manager to try again
    const { DependencyManager } = require('./dependency-manager');
    const depManager = new DependencyManager(this.log);
    
    try {
      await depManager.setupDependencies();
      await dialog.showMessageBox({
        type: 'info',
        title: 'Installation Complete',
        message: 'Dependencies installed successfully!',
        detail: 'Whisper Transcribe is now ready to use.',
        buttons: ['OK']
      });
      app.relaunch();
    } catch (error) {
      await this.showErrorWithRecovery(error, 'Auto Installation');
    }
  }

  async showManualInstallInstructions() {
    const instructions = `# Manual Installation Instructions

Open Terminal and run these commands:

1. Update pip:
   pip3 install --upgrade pip

2. Install required packages:
   pip3 install --user flask flask-cors openai-whisper scipy numpy psutil sounddevice

3. Restart Whisper Transcribe

## If you get permission errors:
Use the --user flag to install to your user directory:
pip3 install --user [package-name]

## Alternative: Create a virtual environment
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install flask flask-cors openai-whisper scipy numpy psutil sounddevice`;

    await dialog.showMessageBox({
      type: 'info',
      title: 'Manual Installation',
      message: 'Manual Installation Instructions',
      detail: instructions,
      buttons: ['Copy Commands', 'OK']
    });
  }

  async showLogFile() {
    const logPath = path.join(app.getPath('userData'), 'whisper-app.log');
    if (fs.existsSync(logPath)) {
      shell.showItemInFolder(logPath);
    } else {
      await dialog.showMessageBox({
        type: 'info',
        title: 'Log File',
        message: 'Log file not found',
        detail: `Log file should be at: ${logPath}`,
        buttons: ['OK']
      });
    }
  }

  async reportIssue(error, context) {
    const issueUrl = 'https://github.com/sethshoultes/whisper-transcribe-pi/issues/new';
    const body = encodeURIComponent(`**Error Context:** ${context}
**Error Message:** ${error.message}
**Stack Trace:**
\`\`\`
${error.stack || 'No stack trace available'}
\`\`\`

**System Info:**
- Platform: ${process.platform}
- Electron: ${process.versions.electron}
- Node: ${process.versions.node}
- App Version: ${app.getVersion()}

**Additional Details:**
[Please describe what you were doing when this error occurred]`);

    shell.openExternal(`${issueUrl}?body=${body}`);
  }

  // Additional helper methods
  async requestPermissions() {
    if (process.platform === 'darwin') {
      await dialog.showMessageBox({
        type: 'info',
        title: 'Grant Permissions',
        message: 'Open System Preferences',
        detail: 'Go to System Preferences → Security & Privacy → Privacy and ensure Whisper Transcribe has the necessary permissions.',
        buttons: ['Open System Preferences', 'OK']
      });
    }
  }

  async showUserInstallInstructions() {
    await dialog.showMessageBox({
      type: 'info',
      title: 'User Directory Installation',
      message: 'Install to User Directory',
      detail: 'Add --user flag to pip commands:\npip3 install --user flask flask-cors openai-whisper scipy numpy psutil sounddevice',
      buttons: ['OK']
    });
  }

  async showPermissionHelp() {
    shell.openExternal('https://support.apple.com/guide/mac-help/control-access-to-files-and-folders-on-mac-mchlp1203/mac');
  }

  async openStorageSettings() {
    if (process.platform === 'darwin') {
      shell.openExternal('x-apple.systempreferences:com.apple.preference.storage');
    }
  }

  async cleanTemporaryFiles() {
    // This could implement actual cleanup logic
    await dialog.showMessageBox({
      type: 'info',
      title: 'Clean Temporary Files',
      message: 'Manual Cleanup Required',
      detail: 'Please manually delete temporary files in:\n• ~/Downloads\n• ~/Library/Caches\n• Empty your Trash',
      buttons: ['OK']
    });
  }

  async showOfflineHelp() {
    await dialog.showMessageBox({
      type: 'info',
      title: 'Offline Mode',
      message: 'Offline Installation',
      detail: 'For offline installation, you can:\n1. Download wheel files on another computer\n2. Transfer them via USB\n3. Install using: pip install [wheel-file]',
      buttons: ['OK']
    });
  }
}

module.exports = { ErrorHandler };