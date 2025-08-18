#!/usr/bin/env node

// Script to create a bundled Python environment for the Electron app
const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');

class BundleCreator {
  constructor() {
    this.appDir = __dirname;
    this.bundleDir = path.join(this.appDir, 'python-bundle');
    this.venvDir = path.join(this.bundleDir, 'venv');
  }

  log(message) {
    console.log(`[Bundle] ${message}`);
  }

  async createBundle() {
    try {
      this.log('Creating bundled Python environment...');
      
      // Clean existing bundle
      if (fs.existsSync(this.bundleDir)) {
        this.log('Removing existing bundle...');
        await this.removeDirectory(this.bundleDir);
      }
      
      // Create bundle directory
      fs.mkdirSync(this.bundleDir, { recursive: true });
      
      // Find Python
      const pythonPath = await this.findPython();
      this.log(`Using Python: ${pythonPath}`);
      
      // Create virtual environment
      await this.createVirtualEnv(pythonPath);
      
      // Install dependencies
      await this.installDependencies();
      
      // Create launcher script
      await this.createLauncher();
      
      // Test the bundle
      await this.testBundle();
      
      this.log('Bundle created successfully!');
      this.log(`Bundle location: ${this.bundleDir}`);
      
    } catch (error) {
      this.log(`Bundle creation failed: ${error.message}`);
      throw error;
    }
  }

  async findPython() {
    const candidates = [
      '/usr/local/bin/python3.11',
      '/usr/local/bin/python3.12',
      '/usr/local/bin/python3.10',
      '/usr/local/bin/python3',
      '/usr/bin/python3',
      'python3.11',
      'python3.12',
      'python3.10',
      'python3',
      'python'
    ];

    for (const candidate of candidates) {
      try {
        const version = await this.execPromise(`${candidate} --version`);
        if (version.includes('Python 3.')) {
          return candidate;
        }
      } catch (error) {
        // Continue to next candidate
      }
    }
    
    throw new Error('No suitable Python found');
  }

  async createVirtualEnv(pythonPath) {
    this.log('Creating virtual environment...');
    await this.execPromise(`${pythonPath} -m venv "${this.venvDir}"`);
  }

  async installDependencies() {
    this.log('Installing dependencies...');
    
    const pipPath = path.join(this.venvDir, 'bin', 'pip');
    const packages = [
      'flask>=2.3.0',
      'flask-cors>=4.0.0',
      'openai-whisper>=20230314',
      'sounddevice>=0.4.6',
      'scipy>=1.10.0',
      'numpy>=1.24.0,<2.0.0',
      'psutil>=5.9.0'
    ];
    
    // Upgrade pip first
    await this.execPromise(`"${pipPath}" install --upgrade pip`);
    
    // Install packages one by one for better error handling
    for (const pkg of packages) {
      this.log(`Installing ${pkg}...`);
      try {
        await this.execPromise(`"${pipPath}" install "${pkg}"`);
      } catch (error) {
        this.log(`Failed to install ${pkg}: ${error.message}`);
        throw error;
      }
    }
  }

  async createLauncher() {
    const pythonExe = path.join(this.venvDir, 'bin', 'python');
    const launcherScript = `#!/bin/bash
# Auto-generated launcher script for bundled Python environment

SCRIPT_DIR="$( cd "$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_EXE="\${SCRIPT_DIR}/venv/bin/python"

if [ ! -f "\$PYTHON_EXE" ]; then
    echo "Error: Python executable not found at \$PYTHON_EXE"
    exit 1
fi

# Run the server with the bundled Python
exec "\$PYTHON_EXE" "\$@"
`;

    const launcherPath = path.join(this.bundleDir, 'python-launcher.sh');
    fs.writeFileSync(launcherPath, launcherScript);
    fs.chmodSync(launcherPath, '755');
    
    this.log(`Created launcher: ${launcherPath}`);
  }

  async testBundle() {
    this.log('Testing bundle...');
    
    const pythonExe = path.join(this.venvDir, 'bin', 'python');
    
    // Test Python works
    await this.execPromise(`"${pythonExe}" --version`);
    
    // Test each package
    const packages = ['flask', 'flask_cors', 'whisper', 'sounddevice', 'scipy', 'numpy', 'psutil'];
    
    for (const pkg of packages) {
      try {
        await this.execPromise(`"${pythonExe}" -c "import ${pkg}; print('${pkg} OK')"`);
        this.log(`✓ ${pkg} imports successfully`);
      } catch (error) {
        throw new Error(`Package ${pkg} failed to import: ${error.message}`);
      }
    }
    
    this.log('All tests passed!');
  }

  async removeDirectory(dirPath) {
    if (fs.existsSync(dirPath)) {
      for (const file of fs.readdirSync(dirPath)) {
        const fullPath = path.join(dirPath, file);
        if (fs.lstatSync(fullPath).isDirectory()) {
          await this.removeDirectory(fullPath);
        } else {
          fs.unlinkSync(fullPath);
        }
      }
      fs.rmdirSync(dirPath);
    }
  }

  execPromise(command) {
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`Command failed: ${command}\n${error.message}\n${stderr}`));
        } else {
          resolve(stdout.trim());
        }
      });
    });
  }
}

// Export for use in other scripts
module.exports = { BundleCreator };

// Run if called directly
if (require.main === module) {
  const creator = new BundleCreator();
  creator.createBundle()
    .then(() => {
      console.log('Bundle creation completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Bundle creation failed:', error.message);
      process.exit(1);
    });
}