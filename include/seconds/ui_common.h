#pragma once
#include "windows.h"
#include <commctrl.h>
#pragma comment(lib, "Comctl32.lib")
#include "stdint.h"
#include "string"
#include "vector"
#include <chrono>

#define COLOR_RED		RGB(196, 84, 84)
#define COLOR_GREEN		RGB(84, 196, 84)
#define COLOR_BLUE		RGB(84, 84, 196)
#define COLOR_ORANGE	RGB(196, 122, 39)
#define COLOR_VIOLET	RGB(148, 76, 161)
#define COLOR_GRAY		0x00E1E1E1
#define COLOR_APP_BG	RGB(240, 240, 240)
#define COLOR_APP_FG	RGB(0, 0, 0)

#define DEF_START_POS	30
#define DEF_HEIGHT		25
#define DEF_LENGTH_ONE	100
#define DEF_LENGTH_TWO	125

constexpr UINT WM_COMPORTS_DONE = WM_APP + 42;

class timer{
private:
    int _enter_time;
    int _timeout;
	bool _active;
public:
    timer();
	void activate(bool on);
    void setTimeout(int timeout);
    bool expired(void);
	void cancel(void);
};


class Button {
public:
	Button() = default;
	void create(HINSTANCE hInst, HWND hparent, HFONT hFont, _In_opt_ LPCWSTR text, int x, int y, int w, int h, int id, int extra = 0);
	~Button() {DeleteObject(hBitmap);}

	void enable(bool enable);
	void setColor(uint32_t color);
	void setName(_In_opt_ LPCWSTR name);
	void setNameAndColor(_In_opt_ LPCWSTR name, uint32_t color);
	void blink(uint32_t color1, uint32_t color2, uint32_t time_ms);
	HWND getHandler(void){return this->hwnd;}
	bool toggle();
	bool Istoggle();


	HWND hwnd;
private:
	WCHAR text[64];
	HFONT hFont;
	HWND hparent;
	uint32_t color;
	HBITMAP hBitmap;
	uint16_t width;
	uint16_t height;

	bool is_toggle = false;
	timer blink_t;
	bool blink_b;
};


class Text {
public:
	Text() = default;
	~Text();
	void create(HINSTANCE hInst, HWND parent, HFONT font, LPCWSTR txt,
		int x, int y, int w, int h, int id, DWORD extraStyle);
	void setText(LPCWSTR text);
	void setFont(HFONT _hfont);
	void setColor(uint32_t color);
	void setBgColor(uint32_t color);
	void setWordWrap(bool wordwrap);
	void setBounds(int x, int y, int w, int h);
	void redrawNow();
	HBRUSH onCtlColorStatic(HDC hdc, uint32_t bg_color);


	enum HAlign { Left, Center, Right };
	enum VAlign { Top, Middle, Bottom };
	void setAlign(HAlign h, VAlign v);
	HWND hwnd;

private:
	WCHAR text[1024];
	HFONT hfont;
	HWND hparent;
	uint16_t width;
	uint16_t height;
	bool word_wrap;
	uint32_t color = COLOR_APP_FG;
	uint32_t bg_color = COLOR_APP_BG;
	HBITMAP hBitmap;
	DWORD extraStyle;
	HBRUSH bgBrush = nullptr;
};

class Edit {
public:
	Edit() = default;
	~Edit() {
		if (hBrush) { DeleteObject(hBrush); hBrush = NULL; }
	}
	void create(HINSTANCE hInst, HWND parent, HFONT hFont, _In_opt_ LPCWSTR text, int x, int y, int w, int h, int id, int extra = 0);
	void enable(bool enable);
	void readonly(bool enable);
	void setColor(uint32_t color);
	void setName(_In_opt_ LPCWSTR name);
	void setNameAndColor(_In_opt_ LPCWSTR name, uint32_t color);
	float getNumber(void) const;
	std::wstring getTextW() const;
	enum HAlign { Left, Center, Right };
	enum VAlign { Top, Middle, Bottom };
	void setAlign(HAlign h, VAlign v);
	LRESULT handlerColor(WPARAM wParam, LPARAM lParam);
	void appendText(_In_opt_ const wchar_t* buffer);
	void clearText();
	HWND hwnd;
private:
	WCHAR text[64];
	HFONT hFont;
	HWND parent;
	int w;
	int h;
	uint32_t color = COLOR_APP_BG;
	uint32_t color_font = COLOR_APP_FG;;
	HBRUSH hBrush;
	HAlign halign_ = Left;
	VAlign valign_ = Top;
	
};

class Tab {
public:
	Tab() = default;
	//~Tab() { hpages.clear(); sel = -1; }

	void create(HINSTANCE hInst, HWND parent, HFONT hFont, int x, int y, int w, int h, int id);
	HWND addPage(const wchar_t* title);
	void select(int idx);
	bool onNotify(LPARAM lParam);
	void resizeToParent(int l, int t, int r, int b);
	void reset(bool destroy_hwnd = true);
	
private:
	HWND hwnd;
	HWND parent;
	HFONT hFont;
	std::vector<HWND> hpages;
	int sel = -1;
	int padl = 0;
	int padt = 0;
	int padr = 0;
	int padb = 0;

	void relayout(void);
};

class Combo {
public:
	Combo() = default;
	void create(HINSTANCE hInst, HWND parent, HFONT hFont, int x, int y, int w, int h, int id);
	void fill(const wchar_t ** list, size_t list_count);
	void fillCOMPorts();
	const wchar_t* getSelected();
	float getSelectedf();
	int getIdxSelected();
	void clear();
	void enable(bool enable);
	HWND hwnd;
private:
	HWND parent;
	HFONT hFont;
	std::wstring cache;
	bool COMIsOpen();
	bool searching = false;
};


class ProgressBar {
public:
	ProgressBar() = default;
	void create(HINSTANCE hInst, HWND parent, HFONT hFont, int x, int y, int w, int h, int total, int step);
	void advance();
	void decrease();
	void setPos(int val);
	int getPos();
	void clear();
	void enable(bool enable);
	HWND hwnd;
private:
	int step;
	int total;
	int pos;
	HWND parent;
};



class Graph {
public:
	Graph() = default;
	struct SeriesStyle {
		COLORREF color = RGB(0, 0, 0);
		std::wstring name;   // para la leyenda
		int penWidth = 2;
	};
	void create(HWND parent, HFONT hFont, int x, int y, int w, int h);
	void setRect(int x, int y, int w, int h);
	RECT getRect() const { return rc; }
	void setSeriesCount(int count);               // crea estilos y buffers
	int  getSeriesCount() const { return (int)series.size(); }
	void setMaxPoints(int maxPoints);             // reserva (no obliga a usar todos)
	int  getMaxPoints() const { return maxPoints; }
	void setSeriesStyle(int idx, const SeriesStyle& st);
	void setSeriesName(int idx, const wchar_t* name);
	void setSeriesColorBGR(int idx, DWORD bgr);   // para tus defines BGR
	void setSeriesColorRGB(int idx, COLORREF rgb);
	void setSeriesData(int idx, const int* values, int n);
	void setDataInterleavedSeriesMajor(const int* values, int n);
	void clear();
	void invalidate(bool erase = FALSE);
	void paint(HDC hdc);
	void paint(HDC hdc, const RECT& rcDraw);
	void setBackground(COLORREF c) { bgColor = c; }
	void setShowLegend(bool v) { showLegend = v; }
	void setShowMinMaxLabels(bool v) { showMinMaxLabels = v; }
	void setGridLines(int lines) { gridLines = (lines < 0) ? 0 : lines; } // nº líneas horizontales
	void setYRangeFixed(bool fixed, int ymin = 0, int ymax = 100);
private:
	HWND parent = nullptr;
	HFONT hFont = nullptr;
	RECT rc{ 0,0,0,0 };
	int maxPoints = 100;
	int nPoints = 1;
	std::vector<SeriesStyle> series;              // estilos por serie
	std::vector<std::vector<int>> y;              // y[serie][punto]
	COLORREF bgColor = RGB(255, 255, 255);
	bool showLegend = true;
	bool showMinMaxLabels = true;
	int gridLines = 3;

	bool yFixed = false;
	int yFixedMin = 0;
	int yFixedMax = 100;

private:
	static int clampi(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }
	static COLORREF BGR2RGB(DWORD bgr);

	void paintImpl(HDC hdc, const RECT& rcDraw);
};

typedef void (*user_func)(void*);

class Thread {
public:
	Thread() = default;
	~Thread() { stop(INFINITE); }

	bool create(user_func fn, uint32_t period_ms, DWORD flags = 0);
	bool stop(DWORD timeout_ms = INFINITE);
	void kill(void);

	DWORD  GetId(void) const { return id; }
	HANDLE GetHandler(void) const { return hthread; }

private:
	static DWORD WINAPI thread_fn(LPVOID lpParam);

private:
	user_func  user_fn = nullptr;
	DWORD      id = 0;
	HANDLE     hevent = nullptr;   // stop event
	HANDLE     hthread = nullptr;
	bool       stopped = true;
	uint32_t   period = 0;         // ms
};


class Slider {
public:
	Slider() = default;

	void create(HINSTANCE hInst, HWND hParent, int x, int y, int w, int h, int id);
	void setRange(int minV, int maxV);
	void setValue(int v, bool redraw = true);
	int  getValue() const { return value; }

	void setGradient(COLORREF left, COLORREF right, bool redraw = true);
	void setBackColor(COLORREF back, bool redraw = true);

	HWND getHandler() const { return hwnd; }

	HWND hwnd = nullptr;

private:
	static ATOM RegisterOnce(HINSTANCE hInst);
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

	void paint(HDC hdc);

	int minV = 0;
	int maxV = 100;
	int value = 50;

	COLORREF gradL = RGB(255, 0, 0);      // rojo
	COLORREF gradR = RGB(128, 0, 255);    // morado
	COLORREF back = RGB(245, 245, 245);  // “vacío” gris claro

	int ctrlId = 0;
};




// Other UI helpers

void destroyAllChildrens(HWND parent);
void centerChildrenOverParent(HWND parent, HWND children);
HWND createChildrenWindow(HWND hparent, HINSTANCE hinst, SUBCLASSPROC subproc, int width, int height, const wchar_t* title);
int createEditDisplay(HWND hwnd, HINSTANCE hinst, HFONT hfont, Edit* edits, size_t count, const wchar_t** names, size_t names_count, const wchar_t** titles, size_t titles_count, int starty, uint8_t ncol);
int createButtonDisplay(HWND hwnd, HINSTANCE hinst, HFONT hfont, Button * buttons, const wchar_t** names, size_t count, int starty, uint8_t ncol, uint16_t start_id = 0);
int createRadioButtonDisplay(HWND hwnd, HINSTANCE hinst, HFONT hfont, Button* buttons, const wchar_t** names, size_t count, int starty, uint8_t ncol, int id);



void updateEditDisplayf(Edit* edits, float* values, wchar_t* fmt, size_t count);
void updateEditDisplayi(Edit* edits, int* values, size_t count);
void updateEditDisplaystr(Edit* edits, wchar_t** values, size_t count);

bool updateButtons(Button* buttons, uint64_t values, size_t count);
