import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronRight, File, Folder, FolderOpen } from 'lucide-react';
import { getCachedData, setCache, CACHE_EXPIRATION, createCacheKey } from '../utils/cacheUtils';

interface FileNode {
  name: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  isOpen?: boolean;
}

interface FileExplorerProps {
  onFileSelect: (filePath: string) => void;
  selectedFile: string;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ onFileSelect, selectedFile }) => {
  const initialFileTree: FileNode[] = [
    {
      name: 'mobile-rust-app',
      type: 'folder',
      isOpen: true,
      children: [
        {
          name: 'src',
          type: 'folder',
          isOpen: true,
          children: [
            { name: 'main.rs', type: 'file' },
            { name: 'lib.rs', type: 'file' },
            { name: 'mobile_utils.rs', type: 'file' },
            { name: 'ui.rs', type: 'file' },
            { name: 'platform.rs', type: 'file' },
          ],
        },
        {
          name: 'platforms',
          type: 'folder',
          isOpen: true,
          children: [
            {
              name: 'android',
              type: 'folder',
              isOpen: false,
              children: [
                { name: 'MainActivity.kt', type: 'file' },
                { name: 'AndroidManifest.xml', type: 'file' },
                { name: 'build.gradle', type: 'file' },
              ],
            },
            {
              name: 'ios',
              type: 'folder',
              isOpen: false,
              children: [
                { name: 'AppDelegate.swift', type: 'file' },
                { name: 'ViewController.swift', type: 'file' },
                { name: 'Info.plist', type: 'file' },
              ],
            },
            {
              name: 'flutter',
              type: 'folder',
              isOpen: false,
              children: [
                { name: 'main.dart', type: 'file' },
                { name: 'pubspec.yaml', type: 'file' },
                { name: 'bridge.rs', type: 'file' },
              ],
            },
          ],
        },
        {
          name: 'tests',
          type: 'folder',
          isOpen: false,
          children: [
            { name: 'integration_test.rs', type: 'file' },
            { name: 'ui_test.rs', type: 'file' },
          ],
        },
        {
          name: 'assets',
          type: 'folder',
          isOpen: false,
          children: [
            { name: 'icons', type: 'folder' },
            { name: 'images', type: 'folder' },
            { name: 'fonts', type: 'folder' },
          ],
        },
        { name: 'Cargo.toml', type: 'file' },
        { name: 'Cargo.lock', type: 'file' },
        { name: 'flutter_rust_bridge.yaml', type: 'file' },
        { name: 'README.md', type: 'file' },
      ],
    },
  ];
  
  const [fileTree, setFileTree] = useState<FileNode[]>(initialFileTree);
  
  // Load file tree from cache on mount and save on update
  useEffect(() => {
    const loadFileTree = async () => {
      const cacheKey = createCacheKey('file-explorer', 'tree');
      const cachedTree = await getCachedData<FileNode[]>(
        cacheKey,
        async () => initialFileTree,
        CACHE_EXPIRATION.VERY_LONG
      );
      
      if (cachedTree) {
        setFileTree(cachedTree);
      }
    };
    
    loadFileTree();
  }, []);
  
  // Save file tree to cache when it changes
  useEffect(() => {
    const saveFileTree = async () => {
      const cacheKey = createCacheKey('file-explorer', 'tree');
      await setCache(cacheKey, fileTree, CACHE_EXPIRATION.VERY_LONG);
    };
    
    saveFileTree();
  }, [fileTree]);

  const toggleFolder = (path: string[]) => {
    const updateNode = (nodes: FileNode[], currentPath: string[]): FileNode[] => {
      return nodes.map((node, index) => {
        if (index === currentPath[0]) {
          if (currentPath.length === 1) {
            return { ...node, isOpen: !node.isOpen };
          } else if (node.children) {
            return {
              ...node,
              children: updateNode(node.children, currentPath.slice(1)),
            };
          }
        }
        return node;
      });
    };

    setFileTree(updateNode(fileTree, path.map(Number)));
  };

  const renderNode = (node: FileNode, path: number[] = [], depth = 0) => {
    const fullPath = [...path, node.name].join('/');
    const isSelected = selectedFile === fullPath;

    return (
      <div key={fullPath}>
        <div
          className={`flex items-center px-2 py-1 hover:bg-gray-700 cursor-pointer text-sm ${
            isSelected ? 'bg-blue-600/30 text-blue-300' : 'text-gray-300'
          }`}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
          onClick={() => {
            if (node.type === 'folder') {
              toggleFolder(path);
            } else {
              onFileSelect(fullPath);
            }
          }}
        >
          {node.type === 'folder' ? (
            <>
              {node.isOpen ? (
                <ChevronDown className="w-4 h-4 mr-1" />
              ) : (
                <ChevronRight className="w-4 h-4 mr-1" />
              )}
              {node.isOpen ? (
                <FolderOpen className="w-4 h-4 mr-2 text-blue-400" />
              ) : (
                <Folder className="w-4 h-4 mr-2 text-blue-400" />
              )}
            </>
          ) : (
            <>
              <div className="w-4 mr-1" />
              <File className="w-4 h-4 mr-2 text-orange-400" />
            </>
          )}
          <span>{node.name}</span>
        </div>
        {node.type === 'folder' && node.isOpen && node.children && (
          <div>
            {node.children.map((child, index) =>
              renderNode(child, [...path, index], depth + 1)
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-800 h-full overflow-y-auto border-r border-gray-700">
      <div className="p-3 border-b border-gray-700">
        <h3 className="text-sm font-medium text-gray-200">Explorer</h3>
      </div>
      <div className="p-1">
        {fileTree.map((node, index) => renderNode(node, [index]))}
      </div>
    </div>
  );
};

export default FileExplorer;