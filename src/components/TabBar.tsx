import React from 'react';
import { X } from 'lucide-react';

interface Tab {
  id: string;
  name: string;
  isDirty: boolean;
}

interface TabBarProps {
  tabs: Tab[];
  activeTab: string;
  onTabSelect: (tabId: string) => void;
  onTabClose: (tabId: string) => void;
}

const TabBar: React.FC<TabBarProps> = ({ tabs, activeTab, onTabSelect, onTabClose }) => {
  return (
    <div className="flex bg-gray-800 border-b border-gray-700 overflow-x-auto">
      {tabs.map((tab) => (
        <div
          key={tab.id}
          className={`flex items-center px-4 py-2 border-r border-gray-700 cursor-pointer min-w-0 group ${
            activeTab === tab.id
              ? 'bg-gray-900 text-white'
              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
          onClick={() => onTabSelect(tab.id)}
        >
          <span className="text-sm truncate mr-2">
            {tab.name}
            {tab.isDirty && <span className="ml-1 text-orange-400">•</span>}
          </span>
          <button
            className="opacity-0 group-hover:opacity-100 hover:bg-gray-600 rounded p-1 transition-opacity"
            onClick={(e) => {
              e.stopPropagation();
              onTabClose(tab.id);
            }}
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      ))}
    </div>
  );
};

export default TabBar;