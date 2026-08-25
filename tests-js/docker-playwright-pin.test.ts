/**
 * Reproducibility contract for the Chromium layer in the published image.
 *
 * The Docker build copies only a subset of workspace manifests before its
 * dependency layer. A Playwright dependency that exists only in a workspace
 * can therefore be absent from that partial install, causing `npx playwright`
 * to fetch whatever version is latest during an otherwise lock-backed build.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

const rootPackage = JSON.parse(
  fs.readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf-8'),
)

const rootLock = JSON.parse(
  fs.readFileSync(path.join(REPO_ROOT, 'package-lock.json'), 'utf-8'),
)

test('Docker Playwright CLI is an exact root dependency with lock integrity', () => {
  const requested = rootPackage.devDependencies?.playwright
  assert.match(
    requested ?? '',
    /^\d+\.\d+\.\d+$/,
    'root playwright must use an exact version (no range)',
  )

  assert.equal(rootLock.packages[''].devDependencies?.playwright, requested)
  const playwright = rootLock.packages['node_modules/playwright']
  assert.equal(playwright?.version, requested)
  assert.match(playwright?.integrity ?? '', /^sha512-/)
  assert.equal(playwright?.dependencies?.['playwright-core'], requested)

  const core = rootLock.packages['node_modules/playwright-core']
  assert.equal(core?.version, requested)
  assert.match(core?.integrity ?? '', /^sha512-/)
})

test('Docker invokes the installed Playwright CLI with ephemeral Node 22', () => {
  const dockerfile = fs.readFileSync(path.join(REPO_ROOT, 'Dockerfile'), 'utf-8')

  assert.match(
    dockerfile,
    /\/opt\/playwright-installer-node\/bin\/node \.\/node_modules\/playwright\/cli\.js\s*\\\s+install --with-deps chromium --only-shell/,
    'Docker must invoke the lock-installed root CLI under the compatible extractor runtime',
  )
  assert.match(
    dockerfile,
    /FROM node:22-bookworm-slim@sha256:[0-9a-f]{64} AS playwright_installer_node/,
    'the extraction-only Node image must be immutable',
  )
  assert.match(
    dockerfile,
    /RUN --mount=from=playwright_installer_node,source=\/usr\/local,target=\/opt\/playwright-installer-node,ro/,
    'Node 22 must be mounted for the install layer instead of copied into the runtime image',
  )
  assert.ok(
    !/\bnpx\s+playwright\s+install\b/.test(dockerfile),
    'npx may fetch the registry latest when the partial workspace install lacks a bin',
  )
})
