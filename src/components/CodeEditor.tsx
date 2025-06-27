import React, { useState, useRef, useEffect } from 'react';
import LiveCursors from './LiveCursors';

interface CodeEditorProps {
  content: string;
  onChange: (content: string) => void;
  language: string;
  fileName: string;
  collaborators?: Array<{
    userId: string;
    userName: string;
    line: number;
    column: number;
    color: string;
  }>;
  onCursorChange?: (line: number, column: number) => void;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ 
  content, 
  onChange, 
  language, 
  fileName, 
  collaborators = [],
  onCursorChange 
}) => {
  const [lineNumbers, setLineNumbers] = useState<number[]>([1]);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const lines = content.split('\n').length;
    setLineNumbers(Array.from({ length: Math.max(lines, 20) }, (_, i) => i + 1));
  }, [content]);

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const lineNumbersEl = document.querySelector('.line-numbers');
    if (lineNumbersEl) {
      lineNumbersEl.scrollTop = e.currentTarget.scrollTop;
    }
  };

  const handleCursorMove = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    const cursorPos = textarea.selectionStart;
    const textBeforeCursor = content.substring(0, cursorPos);
    const lines = textBeforeCursor.split('\n');
    const line = lines.length;
    const column = lines[lines.length - 1].length + 1;
    
    setCursorPosition({ line, column });
    onCursorChange?.(line, column);
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    handleCursorMove(e);
  };

  return (
    <div className="flex h-full bg-gray-900 text-gray-100 font-mono text-sm">
      <div className="line-numbers bg-gray-800 px-4 py-4 text-gray-500 text-right select-none overflow-hidden border-r border-gray-700 min-w-[4rem]">
        {lineNumbers.map((num) => (
          <div key={num} className="leading-6">
            {num}
          </div>
        ))}
      </div>
      <div className="flex-1 relative">
        <LiveCursors cursors={collaborators} currentFile={fileName} />
        <textarea
          ref={textareaRef}
          className="w-full h-full p-4 bg-transparent resize-none outline-none font-mono text-sm leading-6 text-white"
          value={content}
          onChange={handleContentChange}
          onSelect={handleCursorMove}
          onScroll={handleScroll}
          spellCheck={false}
          placeholder={`Start coding in ${fileName}...`}
          style={{
            tabSize: 4,
          }}
        />
      </div>
    </div>
  );
};

export default CodeEditor;